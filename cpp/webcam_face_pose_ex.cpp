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


#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

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

    	
        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
		
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            //printf("rows are %d\n", temp.rows);//cols=640,rows=460
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes2d;
            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = pose_model(cimg,faces[i]);//to pose model den einai function!
                std::vector<cv::Point2d> image_points;
                image_points.push_back(shape.part(36));    // Nose tip
                image_points.push_back(shape.part(39));    // Chin
                image_points.push_back(shape.part(42));     // Left eye left corner
                image_points.push_back(shape.part(45));    // Right eye right corner
                image_points.push_back(shape.part(48));    // Left Mouth corner
                image_points.push_back(shape.part(54));    // Right mouth corner
                //tha xreiastoume ta 6 parakatw 2d landmarks(se pixels):
                //n.36: deksia akri deksiou matiou
                //n.39: aristeri akri deksiou matiou
                //n.42: deksia akri aristerou matiou
                //n.45: aristeri akri aristerou matiou
                //n.48: deksia akri stoma
                //n.54: aristeri akri stoma                                
                cout << "pixel position of 1st part: " << shape.part(0) << endl;
                shapes2d.push_back(pose_model(cimg, faces[i]));//size=68, shape.part(0)         


		//Camera internals. rows=460, cols=640
    		double focal_length = temp.cols;// Approximate focal length.
   		cv::Point2d center = cv::Point2d(temp.cols/2,temp.rows/2);
    		cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    		cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type);// Assuming no lens distortion
		cout << "Camera Matrix " << endl << camera_matrix << endl ;
    	
    		// Output rotation and translation
    		cv::Mat rotation_vector; // Rotation in axis-angle form
    		cv::Mat translation_vector;	
		// Solve for pose
    		cv::solvePnP(model3Dpoints, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector); 


            }
            

            //Now let's view our face poses on the screen.
            /*
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes2d)); 
	    */
	    
        }
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

	

/*
    	vector<Point3d> nose_end_point3D;//Project a 3D point (0,0,1000.0) onto the image plane.
    	vector<Point2d> nose_end_point2D;
    	nose_end_point3D.push_back(Point3d(0,0,1000.0));
     
    	projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
*/



//dlib::array<array2d<rgb_pixel> > face_chips;
//extract_image_chips(cimg, get_face_chip_details(shapes), face_chips);
//win.set_image(tile_images(face_chips));	
        
