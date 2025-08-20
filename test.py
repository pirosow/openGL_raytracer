import turtle

t1 = turtle.Turtle()

center = [0, 0]
angles = 0
zoom = 2

def goto(pos):
    t1.penup()

    t1.goto(pos[0], pos[1])

    t1.pendown()

goto(center)

t1.dot(10, "green")

turtle.done()