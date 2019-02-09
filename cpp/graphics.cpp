/* for the installation, eg. follow the instructions from here:
https://www.youtube.com/watch?v=XnsYl4RvEwo
 */
#include<iostream>
#include<stdio.h>
#include<graphics.h>
using namespace std;

int main()
{
    int gd=DETECT,gm;
    initgraph(&gd,&gm,NULL);
    
    setbkcolor(5);
    cleardevice();
    circle(getmaxx()/2,getmaxy()/2,100);
    circle(getmaxx()/2,getmaxy()/2,120);
    circle(getmaxx()/2,getmaxy()/1,140);
    outtextxy(getmaxx()/2-40,getmaxy()/2,"virtualoops");
    delay(100000);
    return 0;
}