#include <chrono>
#include <future>
#include <numeric>
#include <string>

#include <plog/Formatters/TxtFormatter.h>
#include <plog/Initializers/ConsoleInitializer.h>
#include <plog/Log.h>
#include <opencv2/opencv.hpp>

#include "gui/application.h"
#include "render.h"
#include "vision/camera.h"
#include "vision/detector.h"
#include "vision/factory.h"

const float OBJ_THRESH = 0.25f;
const float SCORE_THRESH = 0.35f;
const float NMS_THRESH = 0.45f;

auto wait_for_frames(vision::Camera& camera) {
    return camera.wait_for_frames();
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
        vision::make_runtime(vision::ModelType::YOLOv8, "yolov12n.onnx",
                             "coco.names", 640, 640, cv::Scalar(114, 114, 114));
    // vision::make_runtime(vision::ModelType::YOLOv5,
    // "yolov5s.onnx",
    //                      "coco.names", 640, 640, cv::Scalar(114,
    //                      114, 114));
    // vision::make_runtime(vision::ModelType::YOLOv8, "RPS-12.onnx",
    //                      "RPS.names", 640, 640, cv::Scalar(114, 114, 114));
    auto detector = vision::Detector(std::move(runtime));
    auto camera = vision::Camera(848, 480, 60);
    const auto thresholds = vision::Thresholds{
        .score = SCORE_THRESH, .nms = NMS_THRESH, .objectness = OBJ_THRESH};

    auto print_fps = [tp_before =
                          std::chrono::steady_clock::now()]() mutable -> void {
        using namespace std::chrono;
        const auto tp_after = steady_clock::now();
        const auto duration = duration_cast<milliseconds>(tp_after - tp_before);
        tp_before = tp_after;

        std::cout << std::fixed << std::setprecision(2)
                  << "fps: " << 1000.f / duration.count() << "\n";
    };

    app.create_video_stream(848, 480);
    app.setVSync(true);

    std::vector<vision::Detection> detections;
    detections.reserve(32);

    auto frames_fut =
        std::async(std::launch::async, wait_for_frames, std::ref(camera));
    while (!app.should_close()) {
        auto frames = frames_fut.get();
        frames_fut =
            std::async(std::launch::async, wait_for_frames, std::ref(camera));
        if (!frames.has_value()) {
            continue;
        }

        if (app.is_inference_enabled()) {
            detector.input(frames->color());
            detector.forward();
            detections = detector.parse(thresholds);
        } else {
            detections.clear();
        }

        static const auto surfaces = std::vector<const cv::Mat*>{
            &frames->color(), &frames->color_depth()};
        static std::size_t surface_index = 0;
        render(camera.depth_scale(), detections, *surfaces[surface_index],
               frames->depth());

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
        app.update_video_stream(frames->color().data,
                                frames->color_depth().data);
        if (const auto depth_picker = app.depth_picker();
            depth_picker.has_value()) {
            const auto distance =
                frames->get_distance(depth_picker->x, depth_picker->y);
            LOG_INFO << distance;
            app.update_depth_picker(distance);
        }
        app.compose_frame();
        app.render();

        app.input();
        // print_fps();
    }

    return 0;
}
