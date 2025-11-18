#include <chrono>
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

// void composeDearImGuiFrame(const Window& window) {
//     ImGui_ImplOpenGL3_NewFrame();
//     ImGui_ImplGlfw_NewFrame();
//
//     ImGui::NewFrame();
//
//     // standard demo window
//     if (show_demo_window) {
//         ImGui::ShowDemoWindow(&show_demo_window);
//     }
//
//     // a window is defined by Begin/End pair
//     {
//         int glfw_width = 0, glfw_height = 0, controls_width = 0;
//         // get the window size as a base for calculating widgets geometry
//         glfwGetFramebufferSize(window.window, &glfw_width, &glfw_height);
//         controls_width = glfw_width;
//         // make controls widget width to be 1/3 of the main window width
//         if ((controls_width /= 3) < 300) {
//             controls_width = 300;
//         }
//
//         // position the controls widget in the top-right corner with some
//         margin ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
//         // here we set the calculated width and also make the height to be
//         // be the height of the main window also with some margin
//         ImGui::SetNextWindowSize(ImVec2(static_cast<float>(controls_width),
//                                         static_cast<float>(glfw_height -
//                                         20)),
//                                  ImGuiCond_Always);
//
//         ImGui::SetNextWindowBgAlpha(0.7f);
//         // create a window and append into it
//         ImGui::Begin("Controls", NULL, ImGuiWindowFlags_NoResize);
//
//         ImGui::Dummy(ImVec2(0.0f, 1.0f));
//         ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Time");
//         // ImGui::Text("%s",
//         // currentTime(std::chrono::system_clock::now()).c_str());
//
//         ImGui::Dummy(ImVec2(0.0f, 3.0f));
//         ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Application");
//         ImGui::Text("Main window width: %d", glfw_width);
//         ImGui::Text("Main window height: %d", glfw_height);
//
//         ImGui::Dummy(ImVec2(0.0f, 3.0f));
//         ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "GLFW");
//         ImGui::Text("%s", glfwGetVersionString());
//
//         ImGui::Dummy(ImVec2(0.0f, 3.0f));
//         ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Dear ImGui");
//         ImGui::Text("%s", IMGUI_VERSION);
//
//         ImGui::Dummy(ImVec2(0.0f, 10.0f));
//         ImGui::Separator();
//         ImGui::Dummy(ImVec2(0.0f, 10.0f));
//
//         // buttons and most other widgets return true when
//         // clicked/edited/activated
//         if (ImGui::Button("Counter button")) {
//             std::cout << "counter button clicked" << std::endl;
//             counter++;
//             if (counter == 9) {
//                 ImGui::OpenPopup("Easter egg");
//             }
//         }
//         ImGui::SameLine();
//         ImGui::Text("counter = %d", counter);
//
//         if (ImGui::BeginPopupModal("Easter egg", NULL)) {
//             ImGui::Text("Ho-ho, you found me!");
//             if (ImGui::Button("Buy Ultimate Orb")) {
//                 ImGui::CloseCurrentPopup();
//             }
//             ImGui::EndPopup();
//         }
//
//         ImGui::Dummy(ImVec2(0.0f, 15.0f));
//         if (!show_demo_window) {
//             if (ImGui::Button("Open standard demo")) {
//                 show_demo_window = true;
//             }
//         }
//
//         // ImGui::Checkbox("show a custom window", &show_another_window);
//         // if (show_another_window) {
//         //     ImGui::SetNextWindowSize(
//         //         ImVec2(250.0f, 150.0f),
//         //         ImGuiCond_FirstUseEver  // after first launch it will use
//         //         values
//         //                                 // from imgui.ini
//         //     );
//         //     // the window will have a closing button that will clear the
//         bool
//         //     // variable
//         //     ImGui::Begin("A custom window", &show_another_window);
//         //
//         //     ImGui::Dummy(ImVec2(0.0f, 1.0f));
//         //     ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Some
//         label");
//         //
//         //     ImGui::TextColored(
//         //         ImVec4(128 / 255.0f, 128 / 255.0f, 128 / 255.0f, 1.0f),
//         "%s",
//         //         "another label");
//         //     ImGui::Dummy(ImVec2(0.0f, 0.5f));
//         //
//         //     ImGui::Dummy(ImVec2(0.0f, 1.0f));
//         //     if (ImGui::Button("Close")) {
//         //         std::cout << "close button clicked" << std::endl;
//         //         show_another_window = false;
//         //     }
//         //
//         //     ImGui::End();
//         // }
//
//         ImGui::End();
//     }
// }

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
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    auto profile = pipe.start(cfg);
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

    rs2::frameset frames = pipe.wait_for_frames();
    while (!app.should_close()) {
        // the frame starts with a clean scene
        // glClearColor(30, 30, 30, 1.0f);
        // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
        //         GL_STENCIL_BUFFER_BIT);
        //
        // // draw our triangle
        // glUseProgram(shaderProgram);
        // // seeing as we only have a single VAO there's no need to bind it
        // every
        // // time, but we'll do so to keep things a bit more organized
        // glBindVertexArray(VAO);
        // glDrawArrays(GL_TRIANGLES, 0, 3);
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

        // detector.input(color_bgr);
        // detector.forward();
        const auto detections = detector.parse(thresholds);

        static const auto surfaces =
            std::vector<const cv::Mat*>{&color_bgr, &depth_rgb};
        static std::size_t surface_index = 0;
        // render(profile, detections, *surfaces[surface_index], depth_z16);

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
        app.update_video_stream(color_bgr.data);
        app.compose_frame();
        app.render();

        app.input();
        print_fps();
    }

    pipe.stop();
    return 0;
}
