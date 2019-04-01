// Example program:
// Using SDL2 to create an application window
//Compile:gcc -Wall -g prog.c -o prog `sdl-config --cflags --libs`
//Compile:$(pkg-config --cflags --libs sdl2)

#include "SDL2/SDL.h"
#include <stdio.h>

int main(int argc, char* argv[]) {

    SDL_Window *window;                    // Declare a pointer
    SDL_Renderer* renderer;
    
    SDL_Init(SDL_INIT_VIDEO);              // Initialize SDL2
    //SDL_SetWindowFullscreen(window,SDL_WINDOW_FULLSCREEN);


    // Create an application window with the following settings:
    window = SDL_CreateWindow(
        "An SDL2 window",                  // window title
        SDL_WINDOWPOS_UNDEFINED,           // initial x position
        SDL_WINDOWPOS_UNDEFINED,           // initial y position
        1240,                               // width, in pixels
        780,                               // height, in pixels
        SDL_WINDOW_OPENGL                 // flags - see below
        //SDL_WINDOW_FULLSCREEN_DESKTOP
    );
    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        printf("Could not create window: %s\n", SDL_GetError());
        return 1;
    }


/************** Color ********************/
     // We must call SDL_CreateRenderer in order for draw calls to affect this window.
    renderer = SDL_CreateRenderer(window, -1, 0);

    // Select the color for drawing. It is set to red here.
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);

    // Clear the entire screen to our selected color.
    SDL_RenderClear(renderer);

    // Up until now everything was drawn behind the scenes.
    // This will show the new, red contents of the window.
    SDL_RenderPresent(renderer);






    // The window is open: could enter program loop here (see SDL_PollEvent())

    SDL_Delay(6000);  // Pause execution for 3000 milliseconds, for example

    // Close and destroy the window
    SDL_DestroyWindow(window);

    // Clean up
    SDL_Quit();
    return 0;
}
