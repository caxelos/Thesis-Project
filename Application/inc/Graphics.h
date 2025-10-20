#include "SDL2/SDL.h"
//#include "SDL2/SDL_ttf.h"
#include "/home/olympia/Downloads/SDL2-2.0.9/include/SDL_ttf.h" 

class Graphics {  
	protected:  
        SDL_Window *window;
        SDL_Renderer *renderer;
        SDL_Rect r[2];
        SDL_Rect r_tmp;
	public:   
		void init(void);
		void setPos(int x_blue,int y_blue,int x_green,int y_green,int x_red, int y_red);
		void close(); 
};