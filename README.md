# ShapeIdentifier
Detect some shapes in an image thanks to opencv.
  - requires **opencv** and **numpy**

![Usage](http://nathsou.fr/iup/u/004d-Capture_du_2015-07-08_17:00:24.png)

#Recognize simple polygons:

  - Green -> Polygon
  - Blue  -> Regular polygon
  - Red   -> Ellipse or Circle

![Before](http://i.imgur.com/z2t854x.png)
![After](http://nathsou.fr/iup/u/ac43-out.png)

if no **-s** *(source image path)* paramater is given, the script will try to capture from a webcam:

![Gameboy](http://nathsou.fr/iup/u/e053-Capture_du_2015-07-08_17:11:06.png)

The default thickness value is -1 *(filled)*, you can change it with the *-t* parameter:
```
python main.py -t 2
```
![RaspberryPi](http://nathsou.fr/iup/u/8c67-Capture_du_2015-07-08_17:20:13.png)
