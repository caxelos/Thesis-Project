#include "SDL2/SDL.h"
#include "SDL2/SDL_ttf.h"
#define RECTANGLE_SIZE 25

class Test {

    public:
        void close() {
          // Close and destroy the window
          SDL_DestroyWindow(this->window);
          //Clean up
          SDL_Quit();
        }
        void init() {
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
        }
                            
        void setPos(int x,int y,bool color) {
            this->r[color].x=x;
            this->r[color].y=y;
            this->r[1].x=x+50;
            this->r[1].y=y+50;
            SDL_SetRenderDrawColor(this->renderer,0,0,255,255);//Blue
            SDL_RenderFillRect(this->renderer, &this->r[color] );
            SDL_SetRenderDrawColor(this->renderer,0,255,0,255);//Green
            SDL_RenderFillRect(this->renderer, &this->r[1] );

        
            SDL_RenderPresent(this->renderer);
            SDL_SetRenderDrawColor(this->renderer, 128,128,128,255);// Select the color for drawing.
            SDL_RenderClear(this->renderer);
        }        
    
    protected:
        SDL_Window *window;
        SDL_Renderer *renderer;
        SDL_Rect r[2];
        SDL_Rect r_tmp;


};

int main() {
    bool quit = false;
    SDL_Event event;
    Test t;
    
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