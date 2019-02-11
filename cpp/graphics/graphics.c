// Example program:
// Using SDL2 to create an application window
//Compile:gcc -Wall -g prog.c -o prog `sdl-config --cflags --libs`
//Compile:$(pkg-config --cflags --libs sdl2)
// if ttf not declared, install:sudo apt-get install libsdl2-ttf-dev
// and 
#include "SDL2/SDL.h"
#include "SDL2/SDL_ttf.h"
#include <stdio.h>

int main(int argc, char* argv[]) {

    bool quit = false;
    SDL_Event event;

    SDL_Window *window;                    // Declare a pointer
    SDL_Renderer* renderer;    
    SDL_Init(SDL_INIT_VIDEO);              // Initialize SDL2
    //SDL_SetWindowFullscreen(window,SDL_WINDOW_FULLSCREEN);

/***************************** Create Window **********************************/

    // Create an application window with the following settings:
    window = SDL_CreateWindow(
        "An SDL2 window",                  // window title
        SDL_WINDOWPOS_UNDEFINED,           // initial x position
        SDL_WINDOWPOS_UNDEFINED,           // initial y position
        1240,                               // width, in pixels
        780,                               // height, in pixels
        SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE// flags - see below
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
    //SDL_RenderPresent(renderer);



/***************** Configure Printable Messages *********************/
    if(TTF_Init()==-1) {
        printf("TTF_Init: %s\n", TTF_GetError());
        exit(2);
    }
    TTF_Font* Sans = TTF_OpenFont("FreeSans.ttf", 24); //this opens a font style and sets a size. 
    if (Sans==NULL) {//Be Carefull here! Always check here for errors!
        printf("Error at font. %s\n",TTF_GetError());
        return 0;
    }
    SDL_Color White = {255, 255, 255};  // this is the color in rgb format, maxing out all would give you the color   white, and it will be your text's color
    SDL_Surface* surfaceMessage = TTF_RenderText_Solid(Sans, "put your text here", White); // as TTF_RenderText_Solid could only be used on SDL_Surface then you have to create the surface first
    SDL_Texture* Message = SDL_CreateTextureFromSurface(renderer, surfaceMessage); //now you can convert it into a texture
    //SDL_Rect Message_rect; //create a rect
    //Message_rect.x = 0;  //controls the rect's x coordinate 
    //Message_rect.y = 0; // controls the rect's y coordinte
    //Message_rect.w = 100; // controls the width of the rect
    //Message_rect.h = 100; // controls the height of the rect
   //Mind you that (0,0) is on the top left of the window/screen, think a rect as the text's box, that way it would be very simple to understance
   //Now since it's a texture, you have to put RenderCopy in your game loop area, the area where the whole code executes
    //SDL_RenderCopy(renderer, Message, NULL, &Message_rect); //you put the renderer's name first, the Message, the crop size(you can ignore this if you don't want to dabble with cropping), and the rect which is the size and coordinate of your texture
    SDL_RenderCopy(renderer, Message, NULL, NULL);
    SDL_RenderPresent(renderer);


    // The window is open: could enter program loop here (see SDL_PollEvent())
    while (!quit)
    {
        SDL_WaitEvent(&event);
 
        switch (event.type)
        {
            case SDL_QUIT:
                quit = true;
                break;
        }
    }
    
    //SDL_Delay(3000);  

    // Close and destroy the window
    SDL_DestroyWindow(window);

    //Clean up
    SDL_DestroyTexture(Message);
    SDL_FreeSurface(surfaceMessage);
    TTF_CloseFont(Sans);
    TTF_Quit();
    SDL_Quit();
    return 0;
}
