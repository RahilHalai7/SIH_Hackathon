import random
import time
import threading
import copy
# pyrefly: ignore [missing-import]
import pygame
import sys

# ============================================================
# CONFIG
# ============================================================

defaultGreen = {0: 10, 1: 10, 2: 10, 3: 10}
defaultRed = 150
defaultYellow = 3

signals = []
noOfSignals = 4
currentGreen = 0
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0

# Guards the shared signal state (currentGreen/currentYellow/signals[*])
# so the timer thread and the render thread never race on it.
signalLock = threading.Lock()

speeds = {'car': 1.4, 'bus': 1.0, 'truck': 1.0, 'bike': 1.8}

directionNumbers = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
directionToIndex = {v: k for k, v in directionNumbers.items()}

# Which direction you end up heading after a turn, relative to the
# direction you were originally travelling in.
turnRightTo = {'right': 'down', 'down': 'left', 'left': 'up', 'up': 'right'}
turnLeftTo = {'right': 'up', 'down': 'right', 'left': 'down', 'up': 'left'}

# Chance a vehicle in the relevant lane turns instead of going straight.
LEFT_TURN_PROB = 0.30   # lane 1 can peel off left
RIGHT_TURN_PROB = 0.30  # lane 2 can peel off right

# How far past the stop line (in the direction of travel) the curve
# starts/ends. TUNE these to match your intersection.png geometry.
turnEntryOffset = 10
turnExitOffset = 60

# Fraction of curve-progress covered per frame; smaller = smoother/slower turn.
TURN_STEP_BASE = 0.028

vehicleTypes = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}

# Spawn coordinates -- TUNE to your images/intersection.png layout.
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
# Frozen copies of the original lane centre-lines, used as the target
# lane position when a vehicle turns into a new road.
BASE_X = copy.deepcopy(x)
BASE_Y = copy.deepcopy(y)

vehicles = {
    'right': {0: [], 1: [], 2: []},
    'down': {0: [], 1: [], 2: []},
    'left': {0: [], 1: [], 2: []},
    'up': {0: [], 1: [], 2: []}
}

signalCoods = [(422, 165), (790, 165), (790, 535), (420, 535)]
signalTimerCoods = [(422, 145), (790, 145), (790, 515), (420, 515)]

trafficStopLines = {'right': 394, 'down': 220, 'left': 811, 'up': 586}
defaultStop = {'right': 386, 'down': 200, 'left': 819, 'up': 594}

stoppingGap = 15
movingGap = 15

pygame.init()
background_temp = pygame.image.load('images/intersection.png')
SCREEN_WIDTH, SCREEN_HEIGHT = background_temp.get_width(), background_temp.get_height()
simulation = pygame.sprite.Group()


class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""


def _quad_bezier(p0, pc, p1, t):
    x_ = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * pc[0] + t ** 2 * p1[0]
    y_ = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * pc[1] + t ** 2 * p1[1]
    return x_, y_


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

        # ---- turning setup ----
        self.willTurn = None
        if lane == 1 and random.random() < LEFT_TURN_PROB:
            self.willTurn = 'left'
        elif lane == 2 and random.random() < RIGHT_TURN_PROB:
            self.willTurn = 'right'
        self.turnPhase = 'approach' if self.willTurn else 'straight'
        self.turnT = 0.0
        self.turnPoints = None  # (entry, control, exit)

        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.image = pygame.image.load(path)
        self.originalImage = self.image

        # stop position relative to the vehicle ahead in the SAME lane
        if (len(vehicles[direction][lane]) > 1 and
                vehicles[direction][lane][self.index - 1].crossed == 0):
            prev = vehicles[direction][lane][self.index - 1]
            if direction == 'right':
                self.stop = prev.stop - prev.image.get_rect().width - stoppingGap
            elif direction == 'left':
                self.stop = prev.stop + prev.image.get_rect().width + stoppingGap
            elif direction == 'down':
                self.stop = prev.stop - prev.image.get_rect().height - stoppingGap
            elif direction == 'up':
                self.stop = prev.stop + prev.image.get_rect().height + stoppingGap
        else:
            self.stop = defaultStop[direction]

        # shift the spawn point back so the next vehicle in this lane queues behind
        if direction == 'right':
            x[direction][lane] -= self.image.get_rect().width + stoppingGap
        elif direction == 'left':
            x[direction][lane] += self.image.get_rect().width + stoppingGap
        elif direction == 'down':
            y[direction][lane] -= self.image.get_rect().height + stoppingGap
        elif direction == 'up':
            y[direction][lane] += self.image.get_rect().height + stoppingGap

        simulation.add(self)

    # ------------------------------------------------------------------
    def _laneImage(self, direction, vehicleClass):
        path = "images/" + direction + "/" + vehicleClass + ".png"
        return pygame.image.load(path)

    def _computeTurnGeometry(self):
        """Build (entry, control, exit) points for a smooth 90-degree curve
        out of the intersection, using the tangent directions at both ends
        so the curve joins cleanly with straight travel on both roads."""
        origin = self.direction
        dest = turnLeftTo[origin] if self.willTurn == 'left' else turnRightTo[origin]

        destLane = self.lane  # reuse same lane index in the new road
        w = self.image.get_rect().width
        h = self.image.get_rect().height

        if origin in ('right', 'left'):
            # travelling horizontally -> will end up travelling vertically
            entryX = trafficStopLines[origin] + (turnEntryOffset if origin == 'right' else -turnEntryOffset)
            entry = (entryX, self.y)

            exitCrossX = BASE_X[dest][destLane]
            exitY = trafficStopLines[dest] + (turnExitOffset if dest == 'down' else -turnExitOffset)
            exit_ = (exitCrossX, exitY)

            control = (exitCrossX, self.y)
        else:
            # travelling vertically -> will end up travelling horizontally
            entryY = trafficStopLines[origin] + (turnEntryOffset if origin == 'down' else -turnEntryOffset)
            entry = (self.x, entryY)

            exitCrossY = BASE_Y[dest][destLane]
            exitX = trafficStopLines[dest] + (turnExitOffset if dest == 'right' else -turnExitOffset)
            exit_ = (exitX, exitCrossY)

            control = (self.x, exitCrossY)

        return entry, control, exit_, dest, destLane

    def _finishTurn(self, dest, destLane):
        """Hand this vehicle over to the destination direction/lane so
        gap-checking against it continues correctly on the new road."""
        self.direction = dest
        self.lane = destLane
        self.crossed = 1
        vehicles[dest][destLane].append(self)
        self.index = len(vehicles[dest][destLane]) - 1
        self.image = self._laneImage(dest, self.vehicleClass)
        self.turnPhase = 'straight'

    # ------------------------------------------------------------------
    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        if self.turnPhase == 'active':
            self._moveTurning()
        else:
            self._moveStraight()

    def _moveTurning(self):
        entry, control, exit_, dest, destLane = self.turnPoints
        step = TURN_STEP_BASE * (self.speed / speeds['car'])
        self.turnT = min(1.0, self.turnT + step)
        px, py = _quad_bezier(entry, control, exit_, self.turnT)
        self.x, self.y = px, py
        if self.turnT >= 1.0:
            self._finishTurn(dest, destLane)

    def _moveStraight(self):
        direction = self.direction
        lane = self.lane
        idx = self.index
        laneVehicles = vehicles[direction][lane]

        with signalLock:
            greenNow = (currentGreen == directionToIndex[direction] and currentYellow == 0)

        if direction == 'right':
            front = self.x + self.image.get_rect().width
            if self.crossed == 0 and front > trafficStopLines['right']:
                self.crossed = 1
                self._maybeStartTurn()
                return
            if self.crossed == 1 or greenNow or front < defaultStop['right']:
                if idx == 0 or front < (laneVehicles[idx - 1].x - movingGap):
                    self.x += self.speed

        elif direction == 'down':
            front = self.y + self.image.get_rect().height
            if self.crossed == 0 and front > trafficStopLines['down']:
                self.crossed = 1
                self._maybeStartTurn()
                return
            if self.crossed == 1 or greenNow or front < self.stop:
                if idx == 0 or front < (laneVehicles[idx - 1].y - movingGap):
                    self.y += self.speed

        elif direction == 'left':
            front = self.x
            if self.crossed == 0 and front < trafficStopLines['left']:
                self.crossed = 1
                self._maybeStartTurn()
                return
            if self.crossed == 1 or greenNow or front > defaultStop['left']:
                if idx == 0 or front > (laneVehicles[idx - 1].x +
                                         laneVehicles[idx - 1].image.get_rect().width + movingGap):
                    self.x -= self.speed

        elif direction == 'up':
            front = self.y
            if self.crossed == 0 and front < trafficStopLines['up']:
                self.crossed = 1
                self._maybeStartTurn()
                return
            if self.crossed == 1 or greenNow or front > defaultStop['up']:
                if idx == 0 or front > (laneVehicles[idx - 1].y +
                                         laneVehicles[idx - 1].image.get_rect().height + movingGap):
                    self.y -= self.speed

    def _maybeStartTurn(self):
        """Called exactly once, the frame a vehicle's front crosses the
        stop line, while its light is green. If it's a turning vehicle,
        kick off the curve; otherwise it just continues straight."""
        if self.willTurn and self.turnPhase == 'approach':
            entry, control, exit_, dest, destLane = self._computeTurnGeometry()
            self.turnPoints = (entry, control, exit_, dest, destLane)
            self.turnT = 0.0
            self.turnPhase = 'active'


# ============================================================
# Signal logic
# ============================================================

def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1])
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[2])
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen[3])
    signals.append(ts4)
    repeat()


def repeat():
    global currentGreen, currentYellow, nextGreen
    while True:
        with signalLock:
            greenLeft = signals[currentGreen].green
        if greenLeft <= 0:
            break
        updateValues()
        time.sleep(1)

    with signalLock:
        currentYellow = 1
    for i in range(0, 3):
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            if vehicle.turnPhase != 'active':
                vehicle.stop = defaultStop[directionNumbers[currentGreen]]

    while True:
        with signalLock:
            yellowLeft = signals[currentGreen].yellow
        if yellowLeft <= 0:
            break
        updateValues()
        time.sleep(1)

    with signalLock:
        currentYellow = 0
        signals[currentGreen].green = defaultGreen[currentGreen]
        signals[currentGreen].yellow = defaultYellow
        signals[currentGreen].red = defaultRed
        currentGreen = nextGreen
        nextGreen = (currentGreen + 1) % noOfSignals
        signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green

    repeat()


def updateValues():
    with signalLock:
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
        dist = [25, 50, 75, 100]
        if temp < dist[0]:
            direction_number = 0
        elif temp < dist[1]:
            direction_number = 1
        elif temp < dist[2]:
            direction_number = 2
        else:
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number])
        time.sleep(1)


def _offScreen(vehicle):
    margin = 100
    return (vehicle.x < -margin or vehicle.x > SCREEN_WIDTH + margin or
            vehicle.y < -margin or vehicle.y > SCREEN_HEIGHT + margin)



class Main:
    thread1 = threading.Thread(name="initialization", target=initialize, args=())
    thread1.daemon = True
    thread1.start()

    background = pygame.image.load('images/intersection.png')
    SCREEN_WIDTH, SCREEN_HEIGHT = background.get_width(), background.get_height()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("SIMULATION")

    black = (0, 0, 0)
    white = (255, 255, 255)

    redSignal = pygame.image.load('images/signals/red.png')
    yellowSignal = pygame.image.load('images/signals/yellow.png')
    greenSignal = pygame.image.load('images/signals/green.png')
    font = pygame.font.Font(None, 30)

    thread2 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())
    thread2.daemon = True
    thread2.start()

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(background, (0, 0))

        with signalLock:
            greenSnapshot = currentGreen
            yellowSnapshot = currentYellow
            signalTexts_data = [(s.red, s.yellow, s.green) for s in signals]

        for i in range(0, noOfSignals):
            if i == greenSnapshot:
                if yellowSnapshot == 1:
                    signals[i].signalText = signalTexts_data[i][1]
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signalTexts_data[i][2]
                    screen.blit(greenSignal, signalCoods[i])
            else:
                red_val = signalTexts_data[i][0]
                signals[i].signalText = red_val if red_val <= 10 else "---"
                screen.blit(redSignal, signalCoods[i])

        for i in range(0, noOfSignals):
            text = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(text, signalTimerCoods[i])

        for vehicle in list(simulation):
            vehicle.render(screen)
            vehicle.move()
            if _offScreen(vehicle):
                simulation.remove(vehicle)  # stop drawing/updating it, keep it in
                                             # vehicles[][] so trailing cars still
                                             # gap-check correctly against its last x/y

        pygame.display.update()
        clock.tick(60)  # fixed frame rate -> speeds/timers behave consistently


Main()
