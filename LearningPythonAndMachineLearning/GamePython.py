import pygame
from pygame import mixer
import random
import math
# Create window game
pygame.init()
mixer.init()
screen = pygame.display.set_mode((700,700))
done = False
x = 60
y = 60    
x_image = random.randint(0,600)
y_image = random.randint(0,600)
#Insert imgae at game
image = pygame.image.load(r'D:\LearningPython\image_1.jpg')
screen.blit(image,(0,0))
#fps
clock = pygame.time.Clock()
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    clock.tick(144)
    # Create icon, name of game
    pygame.display.set_caption("Game")
    icon = pygame.image.load('game.png')
    pygame.display.set_icon(icon)
    #change back Ground
    screen.fill([120,0,0])
    #Draw the rectangle
    #pygame.draw.rect: screen, three number: color
    #pygame.Rect: two number first: point, two number second: weight, height
    # pygame.draw.rect(screen,(0,0,100),pygame.Rect(x,y,100,100))
    pygame.draw.rect(screen,(255,255,255),pygame.Rect(x_image,y_image,100,100))
    #Control
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_UP]:
        y-= 1
    if pressed[pygame.K_DOWN]:
        y += 1
    if pressed[pygame.K_LEFT]:
        x -= 1
    if pressed[pygame.K_RIGHT]:
        x += 1
    #limit
    if x<=0:
        x=0
    elif x>=600:
        x=600
    if y>=600:
         y=600
    elif y<=0:
         y=0
    if x == x_image or y == y_image:
        x_image = random.randint(0,600)
        y_image = random.randint(0,600)
    pygame.display.flip()