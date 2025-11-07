#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <librealsense2/rs.hpp>

#include "dnn_models/yolov5.h"

static const float OBJ_THRESH = 0.25f;
static const float SCORE_THRESH = 0.35f;  // obj * class
static const float NMS_THRESH = 0.45f;

float get_depth_scale(const rs2::pipeline_profile& profile) {
    auto depth_scale = 0.f;

    const auto sensors = profile.get_device().query_sensors();
    for (const auto& s : sensors) {
        if (auto ds = s.as<rs2::depth_sensor>()) {
            depth_scale = ds.get_depth_scale();
            break;
        }
    }

    if (depth_scale <= 0.f) {
        std::cerr << "Failed to get depth scale; defaulting to 0.001\n";
        depth_scale = 0.001f;
    }

    return depth_scale;
}

float get_median_depth(const cv::Mat& depth_z16, const cv::Rect& roi,
                       float depth_scale) {
    cv::Rect clipped = roi & cv::Rect(0, 0, depth_z16.cols, depth_z16.rows);
    if (clipped.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<uint16_t> vals;
    vals.reserve(clipped.area());
    for (int y = clipped.y; y < clipped.y + clipped.height; ++y) {
        const auto* const row = depth_z16.ptr<uint16_t>(y);
        for (int x = clipped.x; x < clipped.x + clipped.width; ++x) {
            const auto d = row[x];
            if (d != 0) {
                vals.push_back(d);
            }
        }
    }

    if (vals.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const std::size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    return vals[mid] * depth_scale;
}

int main() {
    const std::string onnx_path = "yolov5s.onnx";
    const std::string names_path = "coco.names";

    auto model = std::make_shared<YOLOv5>(onnx_path, names_path);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    auto profile = pipe.start(cfg);
    auto depth_scale = get_depth_scale(profile);

    while (true) {
        // GRAB FRAMES
        auto frames = pipe.wait_for_frames();
        auto color = frames.get_color_frame();
        auto depth = frames.get_depth_frame();

        if (!color || !depth) {
            continue;
        }

        const auto color_bgr =
            cv::Mat(color.get_height(), color.get_width(), CV_8UC3,
                    (void*)color.get_data(), cv::Mat::AUTO_STEP);
        const auto depth_z16 =
            cv::Mat(depth.get_height(), depth.get_width(), CV_16U,
                    (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        // PREPROCESS
        model->input(color_bgr);
        model->forward();
        cv::Mat out = model->outputs();

        // PARSE DETECTIONS
        const int rows = out.size[1];  // N
        const int dims = out.size[2];  // 85
        if (dims != 85) {
            std::cerr << "Unexpected output dims=" << dims
                      << " (expected 85)\n";
        }

        std::vector<int> class_ids;
        std::vector<float> scores;
        std::vector<cv::Rect> boxes;

        // data format [x, y, w, h, objectness, probs...(80)]
        auto data = (float*)out.data;
        for (int i = 0; i < rows; ++i, data += dims) {
            float obj = data[4];
            if (obj < OBJ_THRESH) {
                continue;
            }

            // Get class with highest score
            cv::Mat scores_row(1, (int)model->class_names.size(), CV_32F,
                               data + 5);
            cv::Point max_class_point;
            double max_class_score;
            cv::minMaxLoc(scores_row, nullptr, &max_class_score, nullptr,
                          &max_class_point);
            float score = obj * (float)max_class_score;
            if (score < SCORE_THRESH) continue;

            float cx = data[0], cy = data[1], w = data[2], h = data[3];

            // Convert from letterboxed input back to original image coords
            float x = cx - w / 2.0f;
            float y = cy - h / 2.0f;
            // remove padding, then scale back
            float x0 =
                (x - model->letterbox.image_x) / model->letterbox.aspect_ratio;
            float y0 =
                (y - model->letterbox.image_y) / model->letterbox.aspect_ratio;
            float x1 = (x + w - model->letterbox.image_x) /
                       model->letterbox.aspect_ratio;
            float y1 = (y + h - model->letterbox.image_y) /
                       model->letterbox.aspect_ratio;

            int ix = std::max(0, (int)std::round(x0));
            int iy = std::max(0, (int)std::round(y0));
            int iw = std::min(color_bgr.cols - ix, (int)std::round(x1 - x0));
            int ih = std::min(color_bgr.rows - iy, (int)std::round(y1 - y0));
            if (iw <= 0 || ih <= 0) continue;

            boxes.emplace_back(ix, iy, iw, ih);
            class_ids.push_back(max_class_point.x);
            scores.push_back(score);
        }

        // NMS
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESH, NMS_THRESH, keep);

        // Draw + depth
        for (int idx : keep) {
            const cv::Rect& box = boxes[idx];
            int cid = class_ids[idx];

            float obj_depth_m = get_median_depth(depth_z16, box, depth_scale);
            std::string depth_str = std::isnan(obj_depth_m)
                                        ? "n/a"
                                        : cv::format("%.2fm", obj_depth_m);

            cv::rectangle(color_bgr, box, cv::Scalar(30, 119, 252), 2);
            std::string label = model->class_names[cid] + " " + depth_str;
            int base;
            cv::Size tsize =
                cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &base);
            int ty = std::max(0, box.y - 8);
            cv::rectangle(color_bgr,
                          cv::Rect(box.x, ty - tsize.height - 6,
                                   tsize.width + 6, tsize.height + 6),
                          cv::Scalar(30, 119, 252), cv::FILLED);
            cv::putText(color_bgr, label, cv::Point(box.x + 3, ty - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255, 255, 255), 2);
            std::cout << model->class_names[cid] << ": " << depth_str << "\n";
        }

        cv::imshow("Color Image", color_bgr);
        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;
    }

    pipe.stop();
    return 0;
}

// #include <iostream>
// #include <librealsense2/rs.hpp>
//
// int main() {
//     rs2::pipeline p;
//
//     // Configure and start the pipeline
//     p.start();
//
//     while (true) {
//         // Block program until frames arrive
//         rs2::frameset frames = p.wait_for_frames();
//
//         // Try to get a frame of a depth image
//         rs2::depth_frame depth = frames.get_depth_frame();
//
//         // Get the depth frame's dimensions
//         float width = depth.get_width();
//         float height = depth.get_height();
//
//         // Query the distance from the camera to the object in the center
//         of the
//         // image
//         float dist_to_center = depth.get_distance(width / 2, height / 2);
//
//         // Print the distance
//         std::cout << "The camera is facing an object " << dist_to_center
//         << " meters away \r";
//     }
// }
// #include <chrono>
// #include <fstream>
// #include <iostream>
// #include <librealsense2/rs.hpp>
// #include <sstream>
//
// // 3rd party header for writing png files
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"
//
// // Helper function for writing metadata to disk as a csv file
// void metadata_to_csv(const rs2::frame& frm, const std::string& filename);
//
// // This sample captures 30 frames and writes the last frame to disk.
// // It can be useful for debugging an embedded system with no display.
// int main(int argc, char* argv[]) try {
//     rs2::colorizer color_map;
//     rs2::pipeline pipe;
//     // rs2::config cfg;
//     // cfg.enable_stream(RS2_STREAM_DEPTH, 1920, 1080, RS2_FORMAT_ANY,
//     0);
//     // cfg.enable_stream(RS2_STREAM_POSE);
//     // cfg.enable_all_streams();
//     pipe.start();
//     // // Capture 30 frames to give autoexposure, etc. a chance to settle
//     // for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();
//     //
//     // const int FRAME_COUNT = 100;
//     // auto t1 = std::chrono::high_resolution_clock::now();
//     // for (auto i = 0; i < FRAME_COUNT; ++i) pipe.wait_for_frames();
//     // auto t2 = std::chrono::high_resolution_clock::now();
//     // auto duration =
//     std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
//     //
//     // std::cout << duration.count() << " ms\n";
//     // auto singleFrameTime = static_cast<double>(duration.count()) /
//     static_cast<double>(FRAME_COUNT);
//     // std::cout << "fps: " << 1000.0 / singleFrameTime << "\n";
//     //
//     // auto frame = pipe.wait_for_frames();
//     // auto vf = frame.as<rs2::video_frame>();
//     // std::cout << "dimensions: " << vf.get_width() << " " <<
//     vf.get_height() << "\n";
//     // return 0;
//
//     for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();
//
//     // Wait for the next set of frames from the camera. Now that
//     autoexposure,
//     // etc. has settled, we will write these to disk
//     for (auto&& frame : pipe.wait_for_frames()) {
//         // We can only save video frames as pngs, so we skip the rest
//         if (auto vf = frame.as<rs2::video_frame>()) {
//             auto stream = frame.get_profile().stream_type();
//             // Use the colorizer to get an rgb image for the depth stream
//             if (vf.is<rs2::depth_frame>()) vf = color_map.process(frame);
//
//             // Write images to disk
//             std::stringstream png_file;
//             png_file << "rs-save-to-disk-output-" <<
//             vf.get_profile().stream_name() << ".png";
//             stbi_write_png(png_file.str().c_str(), vf.get_width(),
//             vf.get_height(), vf.get_bytes_per_pixel(),
//                            vf.get_data(), vf.get_stride_in_bytes());
//             std::cout << "Saved " << png_file.str() << std::endl;
//
//             // Record per-frame metadata for UVC streams
//             std::stringstream csv_file;
//             csv_file << "rs-save-to-disk-output-" <<
//             vf.get_profile().stream_name() << "-metadata.csv";
//             metadata_to_csv(vf, csv_file.str());
//         }
//     }
//
//     return EXIT_SUCCESS;
// } catch (const rs2::error& e) {
//     std::cerr << "RealSense error calling " << e.get_failed_function() <<
//     "("
//     << e.get_failed_args() << "):\n    "
//               << e.what() << std::endl;
//     return EXIT_FAILURE;
// } catch (const std::exception& e) {
//     std::cerr << e.what() << std::endl;
//     return EXIT_FAILURE;
// }
//
// void metadata_to_csv(const rs2::frame& frm, const std::string& filename)
// {
//     std::ofstream csv;
//
//     csv.open(filename);
//
//     //    std::cout << "Writing metadata to " << filename << endl;
//     csv << "Stream," <<
//     rs2_stream_to_string(frm.get_profile().stream_type())
//     << "\nMetadata Attribute,Value\n";
//
//     // Record all the available metadata attributes
//     for (size_t i = 0; i < RS2_FRAME_METADATA_COUNT; i++) {
//         if (frm.supports_frame_metadata((rs2_frame_metadata_value)i)) {
//             csv <<
//             rs2_frame_metadata_to_string((rs2_frame_metadata_value)i)
//             << ","
//                 << frm.get_frame_metadata((rs2_frame_metadata_value)i) <<
//                 "\n";
//         }
//     }
//
//     csv.close();
// }
