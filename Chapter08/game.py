#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import collections
import numpy as np
import pygame
import random

class MyGame(object):
    
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(10, 100)
        
        # set constants
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_BLACK = (0, 0, 0)
        self.GAME_WIDTH = 400
        self.GAME_HEIGHT = 400
        self.BALL_WIDTH = 20
        self.BALL_HEIGHT = 20
        self.PADDLE_WIDTH = 50
        self.PADDLE_HEIGHT = 10
        self.GAME_FLOOR = 350
        self.GAME_CEILING = 10
        self.BALL_VELOCITY = 10
        self.PADDLE_VELOCITY = 20
        self.FONT_SIZE = 30
        self.MAX_TRIES_PER_GAME = 100
        self.CUSTOM_EVENT = pygame.USEREVENT + 1
        self.font = pygame.font.SysFont("Comic Sans MS", self.FONT_SIZE)
        
        # we want to start the game remotely, so moving this block
        # to its own function
        self.frames = collections.deque(maxlen=4)
        self.game_over = False
        # initialize positions
        self.paddle_x = self.GAME_WIDTH // 2
        self.game_score = 0
        self.ball_x = random.randint(0, self.GAME_WIDTH)
        self.ball_y = self.GAME_CEILING
        self.num_tries = 0
        # set up display, clock, etc
        self.screen = pygame.display.set_mode(
                (self.GAME_WIDTH, self.GAME_HEIGHT))
        self.clock = pygame.time.Clock()
        
        
    def play(self):
        
        # game loop
        while not self.game_over:
            if self.num_tries > self.MAX_TRIES_PER_GAME:
                self.game_over = True
                break
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.game_over = True
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.paddle_x -= self.PADDLE_VELOCITY
                    if self.paddle_x < 0:
                        self.paddle_x = 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    self.paddle_x += self.PADDLE_VELOCITY
                    if self.paddle_x > self.GAME_WIDTH - self.PADDLE_WIDTH:
                        self.paddle_x = self.GAME_WIDTH - self.PADDLE_WIDTH
                    
            self.screen.fill(self.COLOR_BLACK)
            score_text = self.font.render("Score: {:d}/{:d}, Ball: {:d}"
                .format(self.game_score, self.MAX_TRIES_PER_GAME,
                        self.num_tries), True, self.COLOR_WHITE)
#            self.screen.blit(score_text, 
#                ((self.GAME_WIDTH - score_text.get_width()) // 2,
#                 (self.GAME_FLOOR + self.FONT_SIZE // 2)))
                
            # update ball position
            self.ball_y += self.BALL_VELOCITY
            ball = pygame.draw.rect(self.screen, self.COLOR_WHITE,
                                    pygame.Rect(self.ball_x, self.ball_y,
                                                self.BALL_WIDTH,
                                                self.BALL_HEIGHT))
            # update paddle position
            paddle = pygame.draw.rect(self.screen, self.COLOR_WHITE,
                                      pygame.Rect(self.paddle_x, 
                                                  self.GAME_FLOOR,
                                                  self.PADDLE_WIDTH,
                                                  self.PADDLE_HEIGHT))
            # check for collision
            if self.ball_y >= self.GAME_FLOOR - self.BALL_WIDTH // 2:
                if ball.colliderect(paddle):
                    self.game_score += 1
                else:
                    self.game_score -= 1
                self.ball_x = random.randint(0, self.GAME_WIDTH)
                self.ball_y = self.GAME_CEILING
                self.num_tries += 1
            
            pygame.display.flip()
            
            # save last 4 frames
            self.frames.append(pygame.surfarray.array2d(self.screen))
            
            self.clock.tick(30)
        # save last 4 frames
        S = np.array(self.frames)
        with open("../data/game-screenshots.npy", "wb") as fscreenshot:
            np.save(fscreenshot, S)

    
    def get_state(self):
        return np.array(list(self.frames))
    
    def set_action(self, action):
        pass
    
if __name__ == "__main__":
    game = MyGame()
    game.play()
