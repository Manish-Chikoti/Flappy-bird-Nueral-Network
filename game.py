import pygame
import neat
import random
import os
import time

# import visualize
import pickle
pygame.init()

pygame.font.init()

# Coding with russ on yt for pygame mask/methodologies
# one problem I have faced is that to chose between sigmoid or TanH function for decision to jump or not jump,
# on recursive trials over training time and converging to optimal performace TanH gave better result over sigmoid
# as TanH has a wider range of values scattered giving us the oppurtunity to set our threshold
# neat documentation https://neat-python.readthedocs.io/en/latest/config_file.html
# intro to NEAT by https://neat-python.readthedocs.io/en/latest/config_file.html

# defining game window dimensions
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 800

# loading required images Birds,Pipes,Background and Base
BIRD_IMGS = [
    pygame.transform.scale2x(
        pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png"))
    )
    for x in range(1, 4)
]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)


class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATIONS = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0  # tilt of the bird
        self.tick_count = 0  # this is used to compute various metrics that are deemed to change when a frame updates
        self.vel = 0  # current velocity of the bird
        self.height = self.y  # initial position of a bird object initiated
        self.img_count = 0  # metric used to animate the bird
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        disturb = self.vel * self.tick_count + 1.5 * self.tick_count**2

        # fine tuning
        if disturb >= 16:
            disturb = 16
        if disturb < 0:
            disturb = -2

        # updating the y coordinate of the bird
        self.y = self.y + disturb

        # defining the tilting of the bird based on its actions i.e..,
        # if its moving up it tilts to MAX_ROTATIONS defined
        # and if its moving down due to gravity we want it to tilt to face 90degrees down
        if disturb < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATIONS:
                self.tilt = self.MAX_ROTATIONS
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # crude way to make the bird look flapping the wings
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[1]
            self.img_count = 0

        # incase of freefalling of bird , we have to make sure it doesnot over tilt and applying
        # correct image and doesnt not make it look like flapping but still going down
        # and continuing the logical following of frames by updating the img_count
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        # snippet to tilt a bird , center is adjusted because default center of a image is topleft of a image
        # but we want the the bird to tilt about its center
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center
        )
        win.blit(rotated_image, new_rect.topleft)

    # defining the mask to use in future to define the collision
    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 200  # gap between the top and bottom pipe
    VEL = 5  # the velocity of the pipe in horizontal direction

    def __init__(self, x):
        self.x = x  # abscissa of the pipe
        self.top = 0  # the coordinates of the top pipe
        self.height = 0  # height of the pipes
        self.bottom = 0  # the bottom coordinates of the pipe
        self.PIPE_TOP = pygame.transform.flip(
            PIPE_IMG, False, True
        )  # image flipped to make the top pipe
        self.PIPE_BOTTOM = PIPE_IMG  # upside pipe to use as the bottom pipe

        self.passed = False  # did we cross it ?
        self.set_height()  # to set height

    # to randomly intialize the pipes the main recipee of the game
    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    # to move the pipes because we are not moving the bird, but rather we are moving the pipes so
    # that it looks like we are in action
    def move(self):
        self.x -= self.VEL

    # method which uses the computed positions of pipes and renders them
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    # !!!Main method!!! to check if we have reached a terminal in our learning
    # that is if we collided , this is done using mask from pygame to
    # give that pixel perfect collision analysis
    def collide(self, bird):
        # preparing the masks of bird and pipes
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        # offset betwen bird and pipes
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # checkign if we have collided, return boolean type
        t_point = bird_mask.overlap(top_mask, top_offset)
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)

        if t_point or b_point:
            return True

        return False


class Base:
    VEL = 5  # velocity of the base should be equal to pipe
    WIDTH = BASE_IMG.get_width()  # fetchig width of the pipe
    IMG = BASE_IMG  # image of the base

    def __init__(self, y):
        self.y = y
        self.x1 = 0  # starting point of the base
        self.x2 = self.WIDTH  # ending point of the base

    # moving the base
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (SCREEN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()


def main(genomes, config):

    #pygame.init()
    win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]
    
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate(
                (
                    bird.y,
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom),
                )
            )

            if output[0] > 0.5:  # we use a tanh activation function so result will be between -1 and 1. if over 0.5 jump
                bird.jump()

        #base.move()
        add_pipe = False
        rem = []

        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win, birds, pipes, base, score)


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    p = neat.Population(
        config
    )  # population will consist of neural networks that will be evolved.

    p.add_reporter(
        neat.StdOutReporter(True)
    )  # to keep track of and display information about the evolution process
    stats = (
        neat.StatisticsReporter()
    )  # collecting various statistics about the evolution process
    p.add_reporter(
        stats
    )  # now collect and report statistics during the evolution process

    winner = p.run(main, 50)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
