// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

//#include <sys/resource.h>
#include <dlib/opencv.h>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include "Regressor.h"
#include "Graphics.h"
#include <stdlib.h>
#include <ctime>

// sudo cmake -DCMAKE_PREFIX_PATH=../libtorch .. && make
//isws thelei "sudo su"
#include <iostream>
#include <memory>

#define TORCH_MODE
//#define REGRESSION_FORESTS

#ifdef TORCH_MODE
#include <torch/script.h> // One-stop header.
#include "Convolutional.h"
#endif


using namespace dlib;
using namespace std;
#define RIGHT 4
#define LEFT 5

volatile int pixW = 200, pixH=200,pixW_n=400, pixH_n=400;
int pixH_truth=100, pixW_truth=100;
volatile bool quit=false;



void GraphicsThread() { 
    
    bool quit = false;
    SDL_Event event;
    Graphics graphics;
 
    while (!quit) {
        SDL_WaitEvent(&event);
        switch (event.type) {
            case SDL_QUIT:
                quit = true;
                break;
        }
        graphics.setPos(pixW,pixH,false);
    }
    graphics.close(); 
}

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(cv::Mat &R)
{
    cv::Mat Rt;
    transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3,3, shouldBeIdentity.type());
     
    return  cv::norm(I, shouldBeIdentity) < 1e-6;
     
}
 
// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
//cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
void rotationMatrixToEulerAngles(cv::Mat &R, float *headpose, int type)
{
    headpose[0] = /*-*/atan2(R.at<double>(0,2),R.at<double>(2,2));//theta
    headpose[1] =/*-*/asin(R.at<double>(1,2));//phi
}


int main(int argc,char **argv)
{
	int timer = 0; srand (time(NULL));
    #ifdef TORCH_MODE
    Convolutional model;//,model2;
    model.load_model(argv[1]);
    //model2.load_model(argv[2]);
    #endif

    try
    {
       
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }
  

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("../../Downloads/shape_predictor_68_face_landmarks.dat") >> pose_model;
    
        //define our 3D face model:
        std::vector<cv::Point3d> model3Dpoints;
        model3Dpoints.push_back(cv::Point3d(-45.097f, -0.48377f, 2.397f));// Right eye: Right Corner
        model3Dpoints.push_back(cv::Point3d(-21.313f, 0.48377f, -2.397f));// Right eye: Left Corner
        model3Dpoints.push_back(cv::Point3d(21.313f, 0.48377f, -2.397f));// Left eye: Right Corner
        model3Dpoints.push_back(cv::Point3d(45.097f, -0.48377f, 2.397f));// Left eye: Left Corner
        model3Dpoints.push_back(cv::Point3d(-26.3f, 68.595f, -9.8608e-32f));// Mouth: Right Corner
        model3Dpoints.push_back(cv::Point3d(26.3f, 68.595f, -9.8608e-32f));//Mouth: Left Corner  

        cv::Mat face3d = (cv::Mat_<float>(3,4) << -45.097, -21.313, 21.313, 45.097 , -0.48377,0.48377, 0.48377, -0.48377, 2.397,-2.397,-2.397,2.397);
        face3d.convertTo(face3d, CV_32FC1);
    
        #ifndef TORCH_MODE
        Regressor regressor;
        regressor.load_model();
        #endif
        //Create thread for graphics
        std::thread t1(GraphicsThread);
        //std::thread t2(TimerThread, &pixW_truth, &pixH_truth);
        SDL_Event event;
        Graphics graphics;
        graphics.init();
        graphics.setPos(pixW,pixH,0,0,0,0);


        // Grab and process frames until the main window is closed by the user.
        #ifdef TORCH_MODE
        while(!quit)
        #endif

        #ifndef TORCH_MODE
        image_window win;
        while(!win.is_closed() && !quit)
        #endif

        {
                    
            // Grab a frame
            cv::Mat temp;
            //cv::flip(temp, temp, 1);
            if (!cap.read(temp))
            {
                break;
            }
          
            cv_image<bgr_pixel> cimg(temp);
        
            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes2d;

            //to for-loop afto ekteleitai mono an uparxoun faces stin eikona
            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = pose_model(cimg,faces[i]);//to pose model den einai function!
                std::vector<cv::Point2d> image_points;


                //we know the (x,y) image landmark positions
                image_points.push_back(cv::Point2d(shape.part(36).x(),shape.part(36).y()));    // Nose tip
                image_points.push_back(cv::Point2d(shape.part(39).x(),shape.part(39).y()));    // Chin
                image_points.push_back(cv::Point2d(shape.part(42).x(),shape.part(42).y()));    // Left eye left corner
                image_points.push_back(cv::Point2d(shape.part(45).x(),shape.part(45).y()));    // Right eye right corner
                image_points.push_back(cv::Point2d(shape.part(48).x(),shape.part(48).y()));    // Left Mouth corner
                image_points.push_back(cv::Point2d(shape.part(54).x(),shape.part(54).y()));    // Right mouth corner
                                         
                shapes2d.push_back(pose_model(cimg, faces[i]));//size=68, shape.part(0)         
                double focal_length = temp.cols;// Approximate focal length.
                cv::Point2d center = cv::Point2d(temp.cols/2,temp.rows/2);//rows=460, cols=640
                cv::Mat Cr = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
                Cr.convertTo(Cr, CV_32FC1);
                cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);// Assuming no lens distortion
                cv::Mat rotation_vector; // Rotation in axis-angle form
                cv::Mat tvec;   
                cv::solvePnP(model3Dpoints, image_points,Cr, dist_coeffs, rotation_vector, tvec);
                //cout << rotation_vector << endl;
                
                cv::Mat Rr;
                cv::Rodrigues(rotation_vector, Rr);Rr.convertTo(Rr, CV_32FC1);
                //cout << "Rr:"<<Rr << endl;
                tvec.convertTo(tvec, CV_32FC1);
                cv::Mat rotface3d;rotface3d.convertTo(rotface3d, CV_32FC1);

                rotface3d = Rr*face3d;//(3x3)x(3x4)
                for (int o=0;o<4;o++) {
                    rotface3d.col(o) = rotface3d.col(o) + tvec.col(0);   
                }
                cv::Mat reye=0.5*(rotface3d.col(0)
                                             +rotface3d.col(1));
                cv::Mat left_eye_center = 0.5*(rotface3d.col(2)
                                             +rotface3d.col(3));

                //cout << "right eye center:" << right_eye_center << endl;

                float z_scale = 600/cv::norm(reye);
                int fx = 960;int fy = 960;int cx = 30;int cy = 18;
                #ifdef TORCH_MODE
                int NWIDTH = 60;
                int NHEIGHT = 36;
                #endif
                #ifdef REGRESSION_FORESTS
                int NWIDTH = 15;
                int NHEIGHT = 9;
                #endif
                cv::Mat Cn = (cv::Mat_<float>(3, 3) << fx,0,cx,0,fy,cy,0,0,1);              
                Cn.convertTo(Cn, CV_32FC1);
                cv::Mat S = (cv::Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, z_scale);//cv::magnitude(cameramidpoints));
                S.convertTo(S, CV_32FC1);
                cv::Point3d hRx = (cv::Point3d)Rr.col(0);
                cv::Point3d forward = ((cv::Point3d)reye/cv::norm(reye));
                cv::Point3d down = forward.cross(hRx);
                down = down/cv::norm((cv::Mat)down, cv::NORM_L2, cv::noArray());

                cv::Point3d right = down.cross(forward);
                right=right/cv::norm((cv::Mat)right, cv::NORM_L2, cv::noArray());
                //cout << (cv::Mat)right << endl;
                cv::Mat R = (cv::Mat_<float>(3, 3) << ((cv::Mat)right).at<double>(0),((cv::Mat)down).at<double>(0),((cv::Mat)forward).at<double>(0)
                                                    , ((cv::Mat)right).at<double>(1),((cv::Mat)down).at<double>(1),((cv::Mat)forward).at<double>(1)
                                                    , ((cv::Mat)right).at<double>(2),((cv::Mat)down).at<double>(2),((cv::Mat)forward).at<double>(2));
                R.convertTo(R, CV_32FC1);
                R=R.t();
                //cout << "Matrix R:" << R << endl;
                cv::Mat M = S * R;M.convertTo(M, CV_32FC1);// rotation_matrix.inv();//.t();
                //cout << "MatrixM:" << M << endl;
                cv::Mat W = Cn * M * Cr.inv();
                //cout << "matrix W:" << W << endl;
                cv::Mat output = cv::Mat::zeros(cv::Size(NWIDTH, NHEIGHT),CV_8U/*CV_32FC1*/); 
                cv::warpPerspective(temp,output, W, output.size());
                
                rotation_vector.convertTo(rotation_vector, CV_32FC1);
             
                cv::Mat Rn = R*Rr;

                float phi =  -atan(Rn.at<float>(0,2)/Rn.at<float>(2,2));//phi
                float theta = asin(Rn.at<float>(1,2));//theta
              	float phi_r =  -atan(Rr.at<float>(0,2)/Rr.at<float>(2,2));//phi
                float theta_r = asin(Rr.at<float>(1,2));//theta
                cout << "phi_r:(" << phi_r* 180.0/M_PI << "," << theta_r* 180.0/M_PI << ")" << endl; 
                cout << "phi_n:(" << phi* 180.0/M_PI << "," << theta* 180.0/M_PI << ")" << endl; 

                //cout << "pose:(" << phi*(180.0/M_PI)<<","<<theta*(180.0/M_PI)<<")"<<endl;
                //cout << "pose1:(" <<theta1* 180.0/M_PI <<","<<phi1* (180.0/M_PI)<<")"<<endl;  
                //cv::flip(output, output, 0);
                cv::Mat output_orig=output;
                cvtColor( output, output, CV_BGR2GRAY );/// Convert to grayscale
                equalizeHist( output, output);/// Apply Histogram Equalization
                cv::flip(output, output, 0);
                cv::flip(output, output, 1);
                #ifdef REGRESSION_FORESTS
                float pose[2]; pose[0] = phi;pose[1]=theta;
                float gaze_n[2],;
                regressor.predict(pose,output.data,gaze_n);
                #endif

                #ifdef TORCH_MODE
                float pose[2];
                float gaze_n[2];
                pose[0]= phi; pose[1]=theta;
                model.predict(output,pose,gaze_n);
                //model2.predict(output,pose,gaze_n2);
                //model3.predict(output,pose,gaze_n3);
                #endif


              
                //cout << "***** start array *****" << endl;
                //for (int q=0;q<36;q++) {
                //    for (int r=0;r<60;r++) {
                //        cout << (int)output.at<unsigned char>(q,r) << ",";                    
                //    }
                //    cout << endl;
                //}
                //cout << endl;
                //TODO:Allakse ton Face-Detector wste na mporei na 
                //pianei kai meros tou proswpou

                /**** Importamt print****/
                //cout << "prediction gaze_n:(" << -gaze_n[0]* 180.0/M_PI << "," << -gaze_n[1]* 180.0/M_PI << ")" << endl;
                //cout << "tvec:" << tvec << endl;
                //cout << "right_eye_center:" << reye << endl;

                //cv::Mat gazeout = R.inv()*(cv::Mat_<float>(3, 1) << gaze[0], gaze[1], 0);
                //cv::Mat gazeout = R.inv()*(cv::Mat_<float>(3, 1) << gaze[0], gaze[1], 0);   
                //cout << "final pred:" <<  gazeout* 180.0/M_PI << endl;
                //ws apostasi mporw na thewrisw to translation_vector me antitheto z-aksona
                                            
                //tan(gaze[0]) = (dx+x)/(-tvec(2))
                //tan(gaze[1])= (dy+tvec(1))/(-tvec(2)) 
                cv::Mat gaze_r = R.inv()*(cv::Mat_<float>(3, 1) << gaze_n[0], gaze_n[1], 0);
                //cv::Mat gaze_r2 = R.inv()*(cv::Mat_<float>(3, 1) << gaze_n2[0], gaze_n2[1], 0);
                //cv::Mat gaze_r3 = R.inv()*(cv::Mat_<float>(3, 1) << gaze_n2[0], gaze_n2[1], 0);
                //gaze_r.at<float>(0,0) = -gaze_r.at<float>(0,0);
                //cout << "gaze is: " << gaze_r << endl;
                //cout << "headpose pose_r:(" << pose[0] * 180.0/M_PI << "," << pose[1] * 180.0/M_PI  << ")" << endl;
                


                /**** SHMANTIKO print****/
                //cout << "gaze_r:(" << gaze_r.at<float>(0,0)* 180.0/M_PI << "," << gaze_r.at<float>(1,0)* 180.0/M_PI << ")" << endl;
                gaze_n[0]=-gaze_n[0];
                //cout << "gaze_n:(" << gaze_n[0]* 180.0/M_PI << "," << gaze_n[1]* 180.0/M_PI << ")" << endl;
                
                //(orizontio,katheto)

                //to dx einai i metatopisi apo to (0,0), diladi aptin kamera
                #define FACTOR 0.3879
                #define MID_WIDTH_PIXEL 650
				float dx,dy;
 
                if (reye.at<float>(0,0) <0 && gaze_r.at<float>(0,0) > 0) {
                    pixW_n = MID_WIDTH_PIXEL + (-(-reye.at<float>(0,0))+ (-reye.at<float>(2,0))*tan(gaze_n[0]))/FACTOR; 
                    pixW = MID_WIDTH_PIXEL + (-(-reye.at<float>(0,0))+ (-reye.at<float>(2,0))*tan(gaze_r.at<float>(0,0)))/FACTOR;
                }
                else if (reye.at<float>(0,0) <0 && gaze_r.at<float>(0,0) < 0) {
                    pixW = MID_WIDTH_PIXEL + (-(-reye.at<float>(0,0))-tan((-gaze_r.at<float>(0,0)))*(-reye.at<float>(2,0)))/FACTOR; 
                    pixW_n = MID_WIDTH_PIXEL + (-(-reye.at<float>(0,0))-tan((-gaze_n[0]))*(-reye.at<float>(2,0)))/FACTOR; 
                }
                else if (reye.at<float>(0,0)>0 && gaze_r.at<float>(0,0) >0) {
                    pixW = MID_WIDTH_PIXEL + (reye.at<float>(0,0)+tan(gaze_r.at<float>(0,0))*(-reye.at<float>(2,0)))/FACTOR;
                    pixW = MID_WIDTH_PIXEL + (reye.at<float>(0,0)+tan(gaze_n[0])*(-reye.at<float>(2,0)))/FACTOR;
                }
                else if (reye.at<float>(0,0)>0 && gaze_r.at<float>(0,0) <0) {
                	pixW = MID_WIDTH_PIXEL - ((-reye.at<float>(2,0))*tan(-gaze_r.at<float>(0,0))-reye.at<float>(0,0))/FACTOR;
                	pixW_n = MID_WIDTH_PIXEL - ((-reye.at<float>(2,0))*tan(-gaze_n[0])-reye.at<float>(0,0))/FACTOR;
                }

                if (reye.at<float>(1,0) <0 && gaze_r.at<float>(1,0) > 0) {
                   
                    pixH = (-(-reye.at<float>(2,0))*tan(gaze_r.at<float>(1,0))+reye.at<float>(1,0))/FACTOR;
                    pixH_n = (-(-reye.at<float>(2,0))*tan(gaze_n[1])+reye.at<float>(1,0))/FACTOR;

                }
                else if (reye.at<float>(1,0) <0 && gaze_r.at<float>(1,0) < 0) {

           			pixH = (-reye.at<float>(1,0) + tan(-gaze_r.at<float>(1,0))* (-reye.at<float>(2,0)))/FACTOR;
                    pixH_n = (-reye.at<float>(1,0) + tan(-gaze_n[1])* (-reye.at<float>(2,0)))/FACTOR;

                }
                else if (reye.at<float>(1,0)>0 && gaze_r.at<float>(1,0)<0) {
                    pixH = ((-reye.at<float>(2,0))*tan(-gaze_r.at<float>(1,0)) -reye.at<float>(1,0))/FACTOR;
                    pixH_n = ((-reye.at<float>(2,0))*tan(-gaze_n[1]) -reye.at<float>(1,0))/FACTOR;
                }
                else {
                   //looking outside the screen!!!
                	cout << "problem" << endl;

                }
                pixH = pixH-20/FACTOR;// here we subtract the distance from camera.Important??.
    
                graphics.setPos(pixW,pixH,pixW_truth,pixH_truth,pixW_n,pixH_n);
                //cv_image<bgr_pixel> cimg2(output_orig);

                cv_image<unsigned char> cimg2(output);
                #ifndef TORCH_MODE
                win.clear_overlay();
                win.set_image(cimg2);
                win.add_overlay(render_face_detections(shapes2d));
                #endif 
            }

            timer++;
            if (timer == 100) {
            	timer = 0;
          
            	/* generate secret number between 100 and 1340: */
  				pixH_truth = std::rand() % 695;
  				/* generate secret number between 100 and 680: */
  				pixW_truth = std::rand() %1255;

            }
             
        }
        #ifndef TORCH_MODE
        regressor.close();
        #endif
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}