#include "SDL2/SDL.h"
class Graphics {  
	protected:  
        SDL_Window *window;
        SDL_Renderer *renderer;
        SDL_Rect r[2];
        SDL_Rect r_tmp;
	public:   
		void init(void);
		void setPos(int x,int y,bool color);
		void close(); 
};