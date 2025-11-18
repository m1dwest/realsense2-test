#include "application.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>
#include <plog/Log.h>

namespace {

static void glfw_error_callback(int error, const char* description) {
    LOG_ERROR << "GLFW error: " << error << ", " << description;
}

static void framebuffer_size_callback(GLFWwindow* window, int width,
                                      int height) {
    glViewport(0, 0, width, height);
}

struct InitError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct GlProgramError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

const char* vert_shader_src =
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";
const char* frag_shader_src =
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

unsigned compile_vert_shader() {
    unsigned int shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shader, 1, &vert_shader_src, NULL);
    glCompileShader(shader);

    int success;
    char log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, log);
        throw GlProgramError{"ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" +
                             std::string{log}};
    }
    return shader;
}

unsigned compile_frag_shader() {
    unsigned int shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader, 1, &frag_shader_src, NULL);
    glCompileShader(shader);

    int success;
    char log[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, log);
        throw GlProgramError{"ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" +
                             std::string{log}};
    }
    return shader;
}

unsigned link_shader_program(unsigned vert_shader, unsigned frag_shader) {
    const auto program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    int success;
    char log[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, log);
        throw GlProgramError{"ERROR::SHADER::PROGRAM::LINKING_FAILED\n" +
                             std::string{log}};
    }

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
    return program;
}

void build_shader_program() {
    const auto vert_shader = compile_vert_shader();
    const auto frag_shader = compile_frag_shader();
    link_shader_program(vert_shader, frag_shader);
}

void init_glfw() {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        const char* log = nullptr;
        glfwGetError(&log);

        auto what = "Couldn't initialize GLFW" +
                    ((log != nullptr) ? ": " + std::string(log) : "");
        throw InitError{std::move(what)};
    }

    glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    glfwWindowHint(GLFW_STENCIL_BITS, 8);

    std::string glsl_version = "";
#ifdef __APPLE__
    glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    // required on Mac OS
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#elif __linux__
    glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#elif _WIN32
    glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef _WIN32
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    float xscale, yscale;
    glfwGetMonitorContentScale(monitor, &xscale, &yscale);
    LOG_INFO << "Monitor scale: " << xscale << "x" << yscale;
    if (xscale > 1 || yscale > 1) {
        highDPIscaleFactor = xscale;
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    }
#elif __APPLE__
    // to prevent 1200x800 from becoming 2400x1600
    // and some other weird resizings
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);
#endif
}

void init_glad() {
    // load all OpenGL function pointers with glad
    // without it not all the OpenGL functions will be available,
    // such as glGetString(GL_RENDERER), and application might just segfault
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw InitError{"Couldn't initialize GLAD"};
    }

    LOG_INFO << "OpenGL renderer: " << glGetString(GL_RENDERER);
    LOG_INFO << "OpenGL from glad " << GLVersion.major << "."
             << GLVersion.minor;
}

gui::Application::Window create_window(int width, int height,
                                       std::string&& title,
                                       bool is_vsync_enabled) {
    gui::Application::Window window;
    // const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    // glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    window.width = width;
    window.height = height;
    window.title = std::move(title);
    window.window = glfwCreateWindow(window.width,   // mode->width,
                                     window.height,  // mode->height,
                                     window.title.c_str(),
                                     NULL,  // monitor
                                     NULL);
    if (!window.window) {
        throw InitError{"Couldn't create a GLFW window"};
    }

    glfwSetWindowPos(window.window, 100, 100);
    glfwSetWindowSizeLimits(window.window,
                            static_cast<int>(900 * window.hiDPIScale),
                            static_cast<int>(500 * window.hiDPIScale),
                            GLFW_DONT_CARE, GLFW_DONT_CARE);

    glfwSetFramebufferSizeCallback(window.window, framebuffer_size_callback);
    glfwMakeContextCurrent(window.window);
    glfwSwapInterval(static_cast<int>(is_vsync_enabled));

    LOG_INFO << "OpenGL from GLFW "
             << glfwGetWindowAttrib(window.window, GLFW_CONTEXT_VERSION_MAJOR)
             << "."
             << glfwGetWindowAttrib(window.window, GLFW_CONTEXT_VERSION_MINOR)
             << std::endl;

    return window;
}

void init_imgui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // ImGuiIO& io = ImGui::GetIO();
    // (void)io;

    // io.Fonts->AddFontFromFileTTF(fontName.c_str(), 24.0f * highDPIScale,
    // NULL,
    //                              NULL);
    // setImGuiStyle(highDPIScale);

    // setup platform/renderer bindings
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        throw InitError{"Couldn't initialize Dear ImGui GLFW implementation"};
    }
    if (!ImGui_ImplOpenGL3_Init()) {
        throw InitError{"Couldn't initialize Dear ImGui OpenGL implementation"};
    }
}

GLuint create_texture(int width, int height) {
    GLuint texture;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

}  // namespace

namespace gui {

Application::~Application() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // optional: de-allocate all resources once they've outlived their
    // purpose
    // glDeleteVertexArrays(1, &VAO);
    // glDeleteBuffers(1, &VBO);
    // glDeleteProgram(shaderProgram);

    if (_window.has_value()) {
        glfwDestroyWindow(_window->window);
    }
    glfwTerminate();
}

bool Application::init(int width, int height, std::string title) {
    try {
        init_glfw();
        LOG_INFO << "GLFW initialized";

        _window =
            create_window(width, height, std::move(title), _is_vsync_enabled);
        LOG_INFO << "GLFW window created";

        init_glad();
        LOG_INFO << "GLAD initialized";

        init_imgui(_window->window);
        LOG_INFO << "Dear ImGui initialized";

        build_shader_program();
        LOG_INFO << "GlProgram initialized";
    } catch (InitError e) {
        LOG_ERROR << "Initialization failed:";
        LOG_ERROR << e.what();
    } catch (GlProgramError e) {
        LOG_ERROR << "GLProgram initialization failed:";
        LOG_ERROR << e.what();
    }

    return true;
}

void Application::create_video_stream(int width, int height) {
    _video_stream = VideoStream{.width = width,
                                .height = height,
                                .texture = create_texture(width, height)};
}

void Application::update_video_stream(unsigned char* data) const {
    glBindTexture(GL_TEXTURE_2D, _video_stream->texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // just to be safe

    glTexSubImage2D(GL_TEXTURE_2D,
                    0,     // mip level
                    0, 0,  // xoffset, yoffset
                    _video_stream->width, _video_stream->height,
                    GL_BGR,  // format of incoming data
                    GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Application::compose_frame() const {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
    ImGui::Begin("RGB viewer", NULL,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_AlwaysAutoResize);

    if (_video_stream.has_value()) {
        // Size of the image in the ImGui window
        ImVec2 imgSize((float)_video_stream->width,
                       (float)_video_stream->height);

        // Many OpenGL backends expect ImTextureID to be the GLuint cast like
        // this:
        ImTextureID texId = (ImTextureID)(intptr_t)_video_stream->texture;

        // Normal (unflipped) UVs:
        ImVec2 uv0(0.0f, 0.0f);
        ImVec2 uv1(1.0f, 1.0f);

        // If your image appears upside-down, flip Y:
        // ImVec2 uv0(0.0f, 1.0f);
        // ImVec2 uv1(1.0f, 0.0f);

        ImGui::Image(texId, imgSize, uv0, uv1);
    } else {
        LOG_ERROR << "Video stream is not initialized";
    }

    ImGui::End();
    ImGui::PopStyleVar();
}

bool Application::should_close() const {
    return glfwWindowShouldClose(_window->window);
}

void Application::render() const {
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(_window->window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(_window->window);
}

void Application::input() const { glfwPollEvents(); }

void Application::setVSync(bool flag) {
    _is_vsync_enabled = flag;
    glfwSwapInterval(static_cast<int>(flag));
}

}  // namespace gui
