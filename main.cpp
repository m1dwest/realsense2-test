#include <string>

#include <opencv2/opencv.hpp>

#include <librealsense2/rs.hpp>

#include "dnn_models/yolov5.h"
#include "render.h"

int main() {
    const std::string onnx_path = "yolov5s.onnx";
    const std::string names_path = "coco.names";

    auto model = std::make_shared<YOLOv5>(onnx_path, names_path);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    auto profile = pipe.start(cfg);

    while (true) {
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

        model->input(color_bgr);
        model->forward();
        const auto objects = model->parse();

        render(profile, objects, color_bgr, depth_z16);
        int k = cv::waitKey(1);
        if (k == 27 || k == 'q') break;
    }

    pipe.stop();
    return 0;
}
