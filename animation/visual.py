from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
import math
import functions as f 

positionCenter1 = [3, -2, -3]  #begin
positionCenter2 = [4, 4, 4]  #end

eulerAngles1 = [-math.pi/2, 2*math.pi/3, math.pi/4]
eulerAngles2 = [math.pi/5, math.pi/4, math.pi/3]

q = []
q1 = []
q2 = []

t = 0
tm = 30

TIMER_ID = 0
TIMER_INTERVAL = 20

animation_ongoing = False


def initialize():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(700, 600)
    glutCreateWindow("Animation")

    glClearColor(0.6, 0.6, 0.6, 1)
    glEnable(GL_DEPTH_TEST)
    
    glLineWidth(1.5) 

def main():
    initialize()
    
    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    if animation_ongoing:
        glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)
    

    
    global q1
    global q2
    
    A = f.Euler2A(eulerAngles1[0], eulerAngles1[1], eulerAngles1[2])
    p, angle = f.AxisAngle(A)
    q1 = f.AngleAxis2Q(p, angle)
    
    A = f.Euler2A(eulerAngles2[0], eulerAngles2[1], eulerAngles2[2])
    p, angle = f.AxisAngle(A)
    q2 = f.AngleAxis2Q(p, angle)
    
    
    glutMainLoop()
    return


def keyboard(key, x, y):
    
    global animation_ongoing
    
    if ord(key) == 27:
        sys.exit(0)
        
    if ord(key) == ord('g'):
        if not animation_ongoing:
            glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)
            animation_ongoing = True
        animation_ongoing = True
    
    if ord(key) == ord('s'):
        animation_ongoing = False
            
            
def timer(value):
    if value != TIMER_ID:
        return
    
    global t
    global tm 
    global animation_ongoing
    global q
    
    t += 0.15
        
    if t >= tm:
        t = 0
        animation_ongoing = False
        return
    
    glutPostRedisplay()
    
    if animation_ongoing:
        glutTimerFunc(TIMER_INTERVAL, timer, TIMER_ID)

def draw_cube(position, angles):
    glPushMatrix()
    
    #glColor3f(1, 1, 0)
    
    glTranslatef(position[0], position[1], position[2])
    
    A = f.Euler2A(angles[0], angles[1], angles[2])
    p, angle = f.AxisAngle(A)
    

    glRotatef(angle/math.pi*180, p[0], p[1], p[2])
    glutSolidCube(1)
   #glutWireCube(1)

    
    draw_axis(2)
    
    glPopMatrix()

def animate():
    global q
    global t
    global tm
    
    glPushMatrix()
    glColor3f(1, 1, 0)
    
    position = []
    
    position.append((1-t/tm)*positionCenter1[0] + (t/tm)*positionCenter2[0])
    position.append((1-t/tm)*positionCenter1[1] + (t/tm)*positionCenter2[1])
    position.append((1-t/tm)*positionCenter1[2] + (t/tm)*positionCenter2[2])

    glTranslatef(position[0], position[1], position[2])
    
    q = f.slerp(q1, q2, tm, t)
    
    p, angle = f.Q2AxisAngle(q)
    
    glRotatef(angle/math.pi*180, p[0], p[1], p[2])

    glutSolidCube(1)
 


    draw_axis(2)
    
    glPopMatrix()
    
def draw_axis(size):
    
    glDisable(GL_LIGHTING)

    glBegin(GL_LINES)
    glColor3f(0.6, 0, 0)
    glVertex3f(0,0,0)
    glVertex3f(size,0,0)
        
    glColor3f(0,0.6,0)
    glVertex3f(0,0,0)
    glVertex3f(0,size,0)
        
    glColor3f(0,0,0.6)
    glVertex3f(0,0,0)
    glVertex3f(0,0,size)
    
    glEnd()
    
    glEnable(GL_LIGHTING)

def display():
    
    global q
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    window_width = 700
    window_height = 600
    
    glViewport(0, 0, window_width, window_height)
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,float( window_width) /  window_height, 1, 25)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(8, 8, 8, 0, 0, 0, 0, 1, 0)
    
    draw_axis(10) 
    
    pos1 = (0.1, 0.1, 0.1, 1.0)
    diffuse1 = (0.8, 0.1, 0.0, 1.0)
    specular1 = (0.0, 0.8, 0.2, 1.0)
   
    pos2 = (0.8, 0.1, 0.1, 1.0)
    diffuse2 = (0.2, 0.5, 0.0, 1.0)
    specular2 = (0.0, 0.1, 0.2, 1.0)

   

    if not animation_ongoing:
        glPushMatrix()
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, pos1)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse1)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular1)
        draw_cube(positionCenter1, eulerAngles1)
        glPopMatrix()

  
     #


   
    



    glPushMatrix()
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, pos2)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse2)
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular2)
    draw_cube(positionCenter2, eulerAngles2) 
    glPopMatrix()

    animate()
    
    glutSwapBuffers()


if __name__ == '__main__': 
    main()