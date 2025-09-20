import random
import time
import threading
import pygame
import sys

# Default values of signal timers
defaultGreen = {0: 5, 1: 5, 2: 5, 3: 5}
defaultRed = 150
defaultYellow = 5

signals = []
noOfSignals = 4
currentGreen = 0   # which signal is green currently
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0  # is yellow signal active?

speeds = {'car': 1.25, 'bus': 0.9, 'truck': 0.9, 'bike': 1.25}

# Spawn coordinates
x = {
    'right': [0, 0, 0],
    'down': [755, 627, 657],
    'left': [1400, 1400, 1400],
    'up': [602, 560, 590]
}
y = {
    'right': [348, 340, 368],
    'down': [0, 0, 0],
    'left': [498, 426, 396],
    'up': [800, 800, 800]
}

vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0},
    'down': {0: [], 1: [], 2: [], 'crossed': 0},
    'left': {0: [], 1: [], 2: [], 'crossed': 0},
    'up': {0: [], 1: [], 2: [], 'crossed': 0}
}
vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}

# Signal image & timer coordinates
signalCoods = [(422, 165), (790, 165), (790, 535), (420, 535)]
signalTimerCoods = [(422, 145), (790, 145), (790, 515), (420, 515)]

# Stop lines (tuned to zebra crossings)
trafficStopLines = {
    'right': 394,   # left → right
    'down': 220,    # top → bottom
    'left': 811,    # right → left
    'up': 586       # bottom → top
}

# Vehicle stop points (a few px before zebra crossing)
defaultStop = {
    'right': 386,
    'down': 200,
    'left': 819,
    'up': 594
}


# Gap between vehicles
stoppingGap = 15
movingGap = 15

pygame.init()
simulation = pygame.sprite.Group()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""

DEBUG_DOWN = True

class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)

        # stop position relative to vehicle ahead
        if (len(vehicles[direction][lane]) > 1 and
                vehicles[direction][lane][self.index - 1].crossed == 0):
            if direction == 'right':
                self.stop = (vehicles[direction][lane][self.index - 1].stop -
                             vehicles[direction][lane][self.index - 1].image.get_rect().width -
                             stoppingGap)
            elif direction == 'left':
                self.stop = (vehicles[direction][lane][self.index - 1].stop +
                             vehicles[direction][lane][self.index - 1].image.get_rect().width +
                             stoppingGap)
            elif direction == 'down':
                self.stop = (vehicles[direction][lane][self.index - 1].stop -
                             vehicles[direction][lane][self.index - 1].image.get_rect().height -
                             stoppingGap)
                print(f"[DEBUG STOP] DOWN vehicle {self.index}: stop={self.stop}, prev_stop={vehicles[direction][lane][self.index - 1].stop}, prev_height={vehicles[direction][lane][self.index - 1].image.get_rect().height}, gap={stoppingGap}")
            elif direction == 'up':
                self.stop = (vehicles[direction][lane][self.index - 1].stop +
                             vehicles[direction][lane][self.index - 1].image.get_rect().height +
                             stoppingGap)
        else:
            self.stop = defaultStop[direction]
            print(f"[DEBUG STOP] DOWN vehicle {self.index} (first): stop={self.stop}, defaultStop={defaultStop[direction]}")

        # shift spawn coordinate
        if direction == 'right':
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] -= temp
        elif direction == 'left':
            temp = self.image.get_rect().width + stoppingGap
            x[direction][lane] += temp
        elif direction == 'down':
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] -= temp
        elif direction == 'up':
            temp = self.image.get_rect().height + stoppingGap
            y[direction][lane] += temp

        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if self.direction == 'right':
            front = self.x + self.image.get_rect().width
            if self.crossed == 0 and front > trafficStopLines['right']:
                self.crossed = 1
            if (self.crossed == 1 or (currentGreen == 0 and currentYellow == 0) or
                    front < defaultStop['right']):
                if self.index == 0 or front < (vehicles[self.direction][self.lane][self.index - 1].x - movingGap):
                    self.x += self.speed

        elif self.direction == 'down':
            front = self.y + self.image.get_rect().height

            # If front crosses stop line, mark as crossed
            if self.crossed == 0 and front > trafficStopLines['down']:
                self.crossed = 1

            # Move if: already crossed OR green light OR haven't reached stop position yet
            if (self.crossed == 1 or 
                (currentGreen == 1 and currentYellow == 0) or 
                front < self.stop):
                if self.index == 0 or front < (vehicles[self.direction][self.lane][self.index - 1].y - movingGap):
                    self.y += self.speed



        elif self.direction == 'left':
            front = self.x
            if self.crossed == 0 and front < trafficStopLines['left']:
                self.crossed = 1
            if (self.crossed == 1 or (currentGreen == 2 and currentYellow == 0) or
                    front > defaultStop['left']):
                if self.index == 0 or front > (vehicles[self.direction][self.lane][self.index - 1].x +
                                               vehicles[self.direction][self.lane][self.index - 1].image.get_rect().width + movingGap):
                    self.x -= self.speed

        elif self.direction == 'up':
            front = self.y
            if self.crossed == 0 and front < trafficStopLines['up']:
                self.crossed = 1
            if (self.crossed == 1 or (currentGreen == 3 and currentYellow == 0) or
                    front > defaultStop['up']):
                if self.index == 0 or front > (vehicles[self.direction][self.lane][self.index - 1].y +
                                               vehicles[self.direction][self.lane][self.index - 1].image.get_rect().height + movingGap):
                    self.y -= self.speed


# Initialization of signals
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green,
                        defaultYellow, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)
    repeat()


def repeat():
    global currentGreen, currentYellow, nextGreen
    while signals[currentGreen].green > 0:
        updateValues()
        time.sleep(1)
    currentYellow = 1
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while signals[currentGreen].yellow > 0:
        updateValues()
        time.sleep(1)
    currentYellow = 0
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
    currentGreen = nextGreen
    nextGreen = (currentGreen + 1) % noOfSignals
    signals[nextGreen].red = (signals[currentGreen].yellow +
                              signals[currentGreen].green)
    repeat()


def updateValues():
    for i in range(0, noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1


def generateVehicles():
    while True:
        vehicle_type = random.randint(0, 3)
        lane_number = random.randint(1, 2)
        temp = random.randint(0, 99)
        direction_number = 0
        dist = [25, 50, 75, 100]
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        elif temp < dist[3]:
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type],
                direction_number, directionNumbers[direction_number])
        time.sleep(1)


class Main:
    thread1 = threading.Thread(
        name="initialization", target=initialize, args=())
    thread1.daemon = True
    thread1.start()

    # Load background to auto-set window size
    background = pygame.image.load('images/intersection.png')
    screenWidth, screenHeight = background.get_width(), background.get_height()
    screen = pygame.display.set_mode((screenWidth, screenHeight))
    pygame.display.set_caption("SIMULATION")

    # Colours
    black = (0, 0, 0)
    white = (255, 255, 255)

    # Signals & font
    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    thread2 = threading.Thread(
        name="generateVehicles", target=generateVehicles, args=())
    thread2.daemon = True
    thread2.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.blit(background, (0, 0))
        for i in range(0, noOfSignals):
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])
        signalTexts = ["", "", "", ""]

        for i in range(0, noOfSignals):
            signalTexts[i] = font.render(
                str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        for vehicle in simulation:
            screen.blit(vehicle.image, [vehicle.x, vehicle.y])
            vehicle.move()
        pygame.display.update()


Main()
