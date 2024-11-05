#include <GLFW/glfw3.h>
#include <iostream>

int main() {
    // Inicjalizacja GLFW
    if (!glfwInit()) {
        std::cerr << "Nie udało się zainicjalizować GLFW" << std::endl;
        return -1;
    }

    // Ustawienie opcji okna
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Tworzenie okna
    GLFWwindow* window = glfwCreateWindow(800, 600, "Moje Okno GLFW", NULL, NULL);
    if (!window) {
        std::cerr << "Nie udało się stworzyć okna" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Ustawienie kontekstu OpenGL
    glfwMakeContextCurrent(window);

    // Pętla renderująca
    while (!glfwWindowShouldClose(window)) {
        // Ustalanie koloru tła
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Wymiana buforów i przetwarzanie zdarzeń
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Sprzątanie i zakończenie
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
