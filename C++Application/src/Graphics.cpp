#include "SDL2/SDL.h"
//#include "SDL2/SDL_ttf.h"
#include "/home/olympia/Downloads/SDL2-2.0.9/include/SDL_ttf.h" 
#include "Graphics.h"
#include <iostream>
using namespace std;


#define RECTANGLE_SIZE 50


void Graphics::close() {
  // Close and destroy the window
  SDL_DestroyWindow(this->window);
  //Clean up
  SDL_Quit();
}
void Graphics::init() {
    SDL_Init(SDL_INIT_VIDEO);
    
    /***************************** Create Window ***************************/
    this->window = SDL_CreateWindow("An SDL2 window",// window title
                              SDL_WINDOWPOS_UNDEFINED,// initial x position
                              SDL_WINDOWPOS_UNDEFINED,// initial y position
                              1240,// width, in pixels
                              780, // height, in pixels
                              SDL_WINDOW_OPENGL|
                              SDL_WINDOW_RESIZABLE|// flags - see below
                              SDL_WINDOW_MAXIMIZED);
    if (this->window == NULL) {// Check that the window was successfully created
        // In the case that the window could not be made...
        printf("Could not create window: %s\n", SDL_GetError());
        return ;
    }


    /***************************** Color *************************************/
    this->renderer = SDL_CreateRenderer(this->window, -1, 0);// We must call SDL_CreateRenderer in order for draw calls to affec
    SDL_SetRenderDrawColor(this->renderer, 128,128,128,255);// Select the color for drawing. It is set to red here.
    SDL_RenderClear(this->renderer);// Clear the entire screen to our selected color.


    /***************** Configure Rectangles *********************/
    this->r[0].w=RECTANGLE_SIZE;
    this->r[0].h=RECTANGLE_SIZE;
    this->r[1].w=RECTANGLE_SIZE;
    this->r[1].h=RECTANGLE_SIZE;
    this->r[2].w=RECTANGLE_SIZE;
    this->r[2].h=RECTANGLE_SIZE;
    //this->r[2].w=RECTANGLE_SIZE;
    //this->r[2].h=RECTANGLE_SIZE;

    //SDL_GetWindowSize(SDL_Window* window,int*w,int* h);
}
                    
//void Graphics::setPos(int x,int y,bool color,int z) {
void Graphics::setPos(int x_blue,int y_blue,int x_green,int y_green, int x_red, int y_red)  {
    //x:pixW
    //y:pixH
    //z:dx0:
    this->r[0].x=x_blue;
    this->r[0].y=y_blue;
    this->r[1].x=x_green;
    this->r[1].y=y_green;
    this->r[2].x=x_red;
    this->r[2].y=y_red;


    SDL_SetRenderDrawColor(this->renderer,0,0,255,255);//Blue
    SDL_RenderFillRect(this->renderer, &this->r[0] );
    SDL_SetRenderDrawColor(this->renderer,0,255,0,255);//Green
    SDL_RenderFillRect(this->renderer, &this->r[1] );
    SDL_SetRenderDrawColor(this->renderer,255,255,0,255);//Red
    SDL_RenderFillRect(this->renderer, &this->r[2] );
  

/* Add legend    
    TTF_Font* Sans = TTF_OpenFont("Sans.ttf", 10); //this opens a font style and sets a size
    SDL_Color Red = {255,0,255};  // this is the color in rgb format, maxing out all would give you the color white, and it will be your text's color
    SDL_Surface* surfaceMessage = TTF_RenderText_Solid(Sans, "ResNet", Red); // as TTF_RenderText_Solid could only be used on SDL_Surface then you have to create the surface first
    SDL_Texture* Message = SDL_CreateTextureFromSurface(this->renderer, surfaceMessage); //now you can convert it into a texture
    SDL_Rect Message_rect; //create a rect
    Message_rect.x = 1000;  //controls the rect's x coordinate 
    Message_rect.y = 0; // controls the rect's y coordinte
    Message_rect.w = 200; // controls the width of the rect
    Message_rect.h = 100; // controls the height of the rect
    //Mind you that (0,0) is on the top left of the window/screen, think a rect as the text's box, that way it would be very simple to understance
    //Now since it's a texture, you have to put RenderCopy in your game loop area, the area where the whole code executes
    int w,h;
    TTF_SizeText(Sans,"ResNet", &w, &h); cout << "width:"<<w<<",height:"<<h;


    SDL_RenderCopy(this->renderer, Message, NULL, &Message_rect); //you put the renderer's name first, the Message, the crop size(you can ignore this if you don't want to dabble with cropping), and the rect which is the size and coordinate of your texture
SDL_FreeSurface(surfaceMessage); SDL_DestroyTexture(Message);

*/

    SDL_RenderPresent(this->renderer);
    SDL_SetRenderDrawColor(this->renderer, 128,128,128,255);// Select the color for drawing.
    SDL_RenderClear(this->renderer);


}        

/*
int main() {
    bool quit = false;
    SDL_Event event;
    Graphics t;
    
    t.init();
    t.setPos(400,400,false);
    SDL_Delay(3000);
    t.setPos(700,700,false);
    SDL_Delay(3000);
    t.setPos(400,400,false);
    
    while (!quit) {
        SDL_WaitEvent(&event);
        switch (event.type) {
            case SDL_QUIT:
                quit = true;
                break;
        }
    }
    t.close(); 
}
*/