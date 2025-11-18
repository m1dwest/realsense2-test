#include <chrono>
#include <numeric>
#include <string>

#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "gui/application.h"
#include "render.h"
#include "vision/detector.h"
#include "vision/factory.h"

const float OBJ_THRESH = 0.25f;
const float SCORE_THRESH = 0.35f;
const float NMS_THRESH = 0.45f;

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
        LOG_WARNING << "Failed to get depth scale; defaulting to 0.001\n";
        depth_scale = 0.001f;
    }

    return depth_scale;
}

int main() {
    plog::init<plog::TxtFormatter>(plog::debug, plog::streamStdOut);

    gui::Application app{};
    if (const auto is_app_ok = app.init(1280, 720, "RealSense Capture");
        !is_app_ok) {
        LOG_ERROR << "Couldn't initialize GUI application";
        return EXIT_FAILURE;
    }

    auto runtime =
        // vision::make_runtime(vision::ModelType::YOLOv8,
        // "yolov12n.onnx",
        //                      "coco.names", 640, 640, cv::Scalar(114,
        //                      114, 114));
        // vision::make_runtime(vision::ModelType::YOLOv5,
        // "yolov5s.onnx",
        //                      "coco.names", 640, 640, cv::Scalar(114,
        //                      114, 114));
        vision::make_runtime(vision::ModelType::YOLOv8, "RPS-12.onnx",
                             "RPS.names", 640, 640, cv::Scalar(114, 114, 114));
    auto detector = vision::Detector(std::move(runtime));
    const auto thresholds = vision::Thresholds{
        .score = SCORE_THRESH, .nms = NMS_THRESH, .objectness = OBJ_THRESH};

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);
    auto profile = pipe.start(cfg);
    const auto depth_scale = get_depth_scale(profile);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    rs2::colorizer colorize;

    auto print_fps = [tp_before =
                          std::chrono::steady_clock::now()]() mutable -> void {
        using namespace std::chrono;
        const auto tp_after = steady_clock::now();
        const auto duration = duration_cast<milliseconds>(tp_after - tp_before);
        tp_before = tp_after;

        std::cout << std::fixed << std::setprecision(2)
                  << "fps: " << 1000.f / duration.count() << "\n";
    };

    app.create_video_stream(640, 480);
    app.setVSync(true);

    rs2::frameset frames = pipe.wait_for_frames();

    std::vector<vision::Detection> detections;
    detections.reserve(32);
    while (!app.should_close()) {
        pipe.poll_for_frames(&frames);

        auto aligned_frames = align_to_color.process(frames);
        auto color = aligned_frames.get_color_frame();
        auto depth = aligned_frames.get_depth_frame();
        auto depth_colorized = colorize.process(depth);

        if (!color || !depth) {
            continue;
        }

        auto color_bgr = cv::Mat(color.get_height(), color.get_width(), CV_8UC3,
                                 (void*)color.get_data(), cv::Mat::AUTO_STEP);
        const auto depth_z16 =
            cv::Mat(depth.get_height(), depth.get_width(), CV_16U,
                    (void*)depth.get_data(), cv::Mat::AUTO_STEP);
        const auto depth_rgb =
            cv::Mat(color.get_height(), color.get_width(), CV_8UC3,
                    (void*)depth_colorized.get_data(), cv::Mat::AUTO_STEP);

        if (app.is_inference_enabled()) {
            detector.input(color_bgr);
            detector.forward();
            detections = detector.parse(thresholds);
        } else {
            detections.clear();
        }

        static const auto surfaces =
            std::vector<const cv::Mat*>{&color_bgr, &depth_rgb};
        static std::size_t surface_index = 0;
        render(depth_scale, detections, *surfaces[surface_index], depth_z16);

        // int k = cv::waitKey(1);
        // if (k == 27 || k == 'q') exit(0);
        // if (k == 'w') {
        //     ++surface_index;
        //     if (surface_index >= surfaces.size()) {
        //         surface_index = 0;
        //     }
        // }
        // if (k == 'c') {
        //     detector.is_nms_class_agnostic = !detector.is_nms_class_agnostic;
        // }
        // cv::cvtColor(color_bgr, color_bgr, cv::COLOR_BGR2RGB);
        if (!color_bgr.isContinuous()) {
            color_bgr = color_bgr.clone();
        }
        app.update_video_stream(color_bgr.data, depth_rgb.data);
        if (const auto depth_picker = app.depth_picker();
            depth_picker.has_value()) {
            const auto distance =
                depth.get_distance(depth_picker->x, depth_picker->y);
            LOG_INFO << distance;
            app.update_depth_picker(distance);
        }
        app.compose_frame();
        app.render();

        app.input();
        // print_fps();
    }

    pipe.stop();
    return 0;
}
