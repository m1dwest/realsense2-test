#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>

namespace gui {

class Application {
   public:
    struct Window {
        int width;
        int height;
        std::string title;
        GLFWwindow* window = nullptr;
        float hiDPIScale;
    };

    struct VideoStream {
        int width;
        int height;
        unsigned texture;
        std::optional<ImVec2> mouse_pos;
        std::optional<ImVec2> mouse_click;
    };

    enum class Stream { Color, Depth, IR, MAX };

    Application() = default;
    ~Application();

    [[nodiscard]] bool init(int width, int height, std::string title);
    void create_video_stream(int width, int height);
    void update_video_stream(unsigned char* color_bgr, unsigned char* depth_rgb,
                             unsigned char* ir_y8) const;
    std::optional<ImVec2> depth_picker() const;
    void update_depth_picker(float depth);
    bool is_inference_enabled() const;
    void compose_frame();
    bool should_close() const;
    void render() const;
    void input() const;

    void setVSync(bool flag);

   private:
    const char* enum_stream_to_cstr(Stream stream) const;

    bool _is_vsync_enabled = true;
    Stream _current_stream = Stream::Color;
    bool _is_inference_enabled = false;

    std::map<Stream, std::string> _stream_map{{Stream::Color, "color"},
                                              {Stream::Depth, "depth"},
                                              {Stream::IR, "infrared"},
                                              {Stream::MAX, "invalid"}};

    std::optional<Window> _window;
    std::optional<VideoStream> _video_stream;
    std::optional<float> _depth_picker;
};

}  // namespace gui
