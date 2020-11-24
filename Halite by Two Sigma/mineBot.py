weights='''1.065318617455976 542.1433864410643 0.7511632555608448 0.6945893010559424 0.1341607259959342 -256.54011220873883
0 2.3837319660395457 0.4770079274532575 14.871982834273645 10
0.04043743652542793 219.09952521708655 9.561641308515489 1.1406984927798645 0.4806089913651024 11.485903586701356
0.32917669267944993 0.12670831197102922
1 -3.1819320805078153 -3
112.69692418951784
3 0.1
5'''
# Contains all dependencies used in bot
# First file loaded

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import math, random
import numpy as np
import scipy.optimize
import scipy.ndimage
from queue import PriorityQueue

# Global constants

    # Infinity value thats actually not infinity
INF = 999999999999
    # All game state goes here - everything, even mundane
state = {}

    # Bot training weights
        # 0 - shipyard reward
        # 1 - mine reward
        # 2 - attack weights
        # 3 - return weights
        # 4 - spawn weights
        # 5 - guard weights
        # 6 - navigation weights
        # 7 - target attack weights
    
temp = []
weights = weights.split('\n')
for line in weights:
    temp.append(np.array(list(map(float,line.split()))))
weights = temp

# Init function - called at the start of each game
def init(board):
    global state
    np.set_printoptions(precision=3)
    state['configuration'] = board.configuration
    state['me'] = board.current_player_id
    state['playerNum'] = len(board.players)
    state['memory'] = {}

    pass

# Run start of every turn
def update(board):
    global action
    action = {}
    state['currentHalite'] = board.current_player.halite
    state['next'] = np.zeros((board.configuration.size,board.configuration.size))
    state['board'] = board
    state['memory'][board.step] = {}
    state['memory'][board.step]['board'] = board
    state['cells'] = board.cells.values()
    state['ships'] = board.ships.values()
    state['myShips'] = board.current_player.ships
    state['shipyards'] = board.shipyards.values()
    state['myShipyards'] = board.current_player.shipyards

    # Calc processes
    encode()
    
    state['spawn'] = spawn()
# General random helper functions that are not strictly "process" or in "nav"

# Map from 0 to 1
def normalize(v):
    norm = np.linalg.norm(v,np.inf)
    if norm == 0: 
       return v
    return v / norm

def closest_ship(t):
    return closest_thing(t,state['myShips'])

def closest_thing(t,arr):
    res = None
    for thing in arr:
        if res == None:
            res = thing
        elif dist(t,res.position) > dist(t,thing.position):
            res = thing
    return res

def closest_thing_position(t,arr):
    res = None
    for thing in arr:
        if res == None:
            res = thing
        elif dist(t,res) > dist(t,thing):
            res = thing
    return res

def halite_per_turn(deposit, shipTime, returnTime):
    travelTime = shipTime + returnTime
    actualDeposit = min(500,deposit * 1.02 ** shipTime)
    maximum = 0
    for turns in range(1,10):
        mined = (1 - .75**turns) * actualDeposit
        perTurn = mined / (turns+travelTime)
        maximum = perTurn if perTurn > maximum else maximum
    return maximum


def miner_num():
    
    if state['board'].step < 280:
        if len(state['myShips']) > 25:
            return min(len(state['myShips']),int(state['haliteMean'] / 4 + len(state['myShipyards'])))
        else:
            return min(len(state['myShips']),int(state['haliteMean'] / 2 + len(state['myShipyards'])))
    elif state['board'].step > 370:
        return len(state['myShips'])
    else:
        return len(state['myShips']) * 0.8

def get_targets():
    targets = []
    for ship in state['enemyShips']:
        if ship.halite != 0:
            targets.append(ship)
    return targets

def attack(ships):
    global action

    # Select potential targets
    targets = get_targets()
    # Greedy selection
    target_list = []
    for ship in ships:
        # Force return
        if ship.halite > 0:
            action[ship] = (INF, ship, state['closestShipyard'][ship.position.x][ship.position.y])
            continue
        # Attack
        finalTarget = targets[0]
        v = rule_attack_reward(ship,finalTarget,target_list)
        for target in targets:
            tv = rule_attack_reward(ship,target,target_list)
            if tv > v:
                v = tv
                finalTarget = target
        target_list.append(finalTarget)
        action[ship] = (1/dist(finalTarget.position,ship.position), ship, finalTarget.position)

# Greedy selection 
# TODO: Improve this!
def rule_attack_reward(s,t,target_list):
    tPos = t.position 
    sPos = s.position
    d = dist(tPos,sPos)
    res = 1/d
    if t.player == state['killTarget']:
        res = res * 4

    control = state['positiveControlMap'][tPos.x][tPos.y]
    if control > 1 and d < 8:
        # Check if local maxima
        yes = True
        for x in range(-3,4):
            if not yes:
                break
            for y in range(-3,4):
                xx = (tPos.x+x) % 21
                yy = (tPos.y+y) % 21
                if not yes:
                    break
                if state['positiveControlMap'][xx][yy] > control and state['enemyShipHalite'][xx][yy] < 99999 and state['enemyShipHalite'][xx][yy] > 0:
                    yes = False
        if yes:
            res = res * 6
    
    if state['trapped'][t.player_id][tPos.x][tPos.y] and d <= 6:
        res = res * 10



    '''
    for pos in get_adjacent(tPos):
        if state['enemyShipHalite'][pos.x][pos.y] <= s.halite:
            return 0
    '''

    return res


###################
# target based attack system
###################

'''
def target_based_attack():
    # actions[ship] = (priority: int, ship: Ship, target: Point)
    params = weights[7] # <- np.array
    # target selection
    targets = "all enemy ships with cargo > 0"
    sorted(targets, key="cargo")

    # assignment
    for target in targets:
        actions["all ally ships with cargo < target.cargo" in area5x5(target)] = ("priority", "ship", "target.pos")
'''


# Core strategy

action = {}  # ship -> (value,ship,target)
farms = [] # list of cells to farm

def farm_tasks():
    build_farm()
    control_farm()
    # Create patrols

def ship_tasks():  # update action
    global action
    cfg = state['configuration']
    board = state['board']
    me = board.current_player
    tasks = {}
    shipsToAssign = []

    # Split attack ships and mine ships
    temp = get_targets()
    state['attackers'] = []
    if len(temp) > 0:
        minerNum = miner_num()
        attackerNum = len(state['myShips']) - minerNum
        for ship in me.ships:
            if ship in action:
                continue
            if attackerNum > 0:
                attackerNum -= 1
                #Uncomment to activate attack
                state['attackers'].append(ship)


    #target_based_attack()

    
    for ship in state['ships']:
        if ship.player_id != state['me']:
            if state['trapped'][ship.player_id][ship.position.x][ship.position.y] and ship.halite > 0:
                print(ship.position)
    

    # All ships rule based
    for ship in me.ships:

        '''
         # Flee
        if state['trapped'][state['me']][ship.position.x][ship.position.y] and ship.halite > 0:
            action[ship] = (INF*2+state[ship]['danger'][ship.position.x][ship.position.y], ship, state['closestShipyard'][ship.position.x][ship.position.y])
        '''
        if ship in action:
            continue 

        for target in get_adjacent(ship.position):
            if board.cells[target].ship != None:
                targetShip = board.cells[target].ship
                if targetShip.player.id != state['me'] and targetShip.halite < ship.halite:
                    action[ship] = (INF*2+state[ship]['danger'][ship.position.x][ship.position.y], ship, state['closestShipyard'][ship.position.x][ship.position.y])

        if ship in action:
            continue # continue its current action

        # End-game return
        if board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite > 0:
            action[ship] = (ship.halite, ship, state['closestShipyard'][ship.position.x][ship.position.y])
        # End game attack
        if len(state['board'].opponents) > 0 and board.step > state['configuration']['episodeSteps'] - cfg.size * 1.5 and ship.halite == 0:
            #print(ship.position)
            if len(state['myShipyards']) > 0 and ship == closest_thing(state['myShipyards'][0].position,state['myShips']):
                action[ship] = (0,ship,state['myShipyards'][0].position)
                continue
            killTarget = state['killTarget']
            if len(killTarget.shipyards) > 0:
                target = closest_thing(ship.position,killTarget.shipyards)
                action[ship] = (ship.halite, ship, target.position)
            elif len(killTarget.ships) > 0:
                target = closest_thing(ship.position,killTarget.ships)
                action[ship] = (ship.halite, ship, target.position)
        
        
        if ship in action or ship in state['attackers']:
            continue

        shipsToAssign.append(ship)

    # Rule based: Attackers
    #print(len(state['myShips']))
    #print(len(state['attackers']))
    attack(state['attackers'])

    # Reward based: Mining + Guarding + Control
    targets = [] # (cell, type)
    for i in board.cells.values():  # Filter targets
        if i.shipyard != None and i.shipyard.player_id == state['me']:
            targets.append((i,'guard'))
            for j in range(min(6,len(state['myShips']))):
                targets.append((i,'cell'))
            continue
        '''if i.halite < 15 and i.ship == None and i.shipyard == None:
            # Spots not very interesting
            continue'''
        if i.ship != None and i.ship.player_id != state['me']:
            if i.ship.halite == 0 and state['controlMap'][i.position.x][i.position.y] < 0:
                continue
        targets.append((i,'cell'))
    rewards = np.zeros((len(shipsToAssign), len(targets)))
    for i, ship in enumerate(shipsToAssign):
        for j, target in enumerate(targets):
            rewards[i, j] = get_reward(ship, target)          
    rows, cols = scipy.optimize.linear_sum_assignment(rewards, maximize=True)  # rows[i] -> cols[i]
    for r, c in zip(rows, cols):
        task = targets[c]
        if task[1] == 'cell':
            cell = cell = targets[c][0]
            if cell.halite == 0 and cell.shipyard == None and (cell.ship == None or cell.ship.player_id == state['me']):
                action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)
            else:
                action[shipsToAssign[r]] = (rewards[r][c], shipsToAssign[r], targets[c][0].position)
        elif task[1] == 'guard':
            action[shipsToAssign[r]] = (0, shipsToAssign[r], targets[c][0].position)
            
    # Process actions
    actions = list(action.values())
    actions.sort(reverse=True, key=lambda x: x[0])
    for act in actions:
        process_action(act)

def process_action(act):
    global action
    if action[act[1]] == True:
        return act[1].next_action
    action[act[1]] = True
    # Processing
    act[1].next_action = d_move(act[1], act[2], state[act[1]]['blocked'])
    # Ship convertion
    sPos = act[1].position
    if state['closestShipyard'][sPos.x][sPos.y] == sPos and state['board'].cells[sPos].shipyard == None:
        act[1].next_action = ShipAction.CONVERT
        state['next'][sPos.x][sPos.y] = 1
    return act[1].next_action

def convert_tasks():
    global action

    # Add convertion tasks

    currentShipyards = state['myShipyards']  # Shipyards "existing"
    targetShipyards = currentShipyards[:]

    # Maximum cell
    v = shipyard_value(state['board'].cells[Point(0,0)])
    t = state['board'].cells[Point(0,0)]
    for cell in state['board'].cells.values():
        a = shipyard_value(cell)
        if v < a:
            v = a
            t = cell
    tx, ty = t.position.x,t.position.y
    # Calculate the reward for each cell
    if state['board'].step == 0:
        # Build immediately
        targetShipyards.append(state['board'].cells[state['myShips'][0].position])
        action[state['myShips'][0]] = (math.inf, state['myShips'][0], state['myShips'][0].position)
        state['currentHalite'] -= 500
    elif len(currentShipyards) == 0:
        # Grab the closest possible ship to the target and build.
        possibleShips = []
        for ship in state['myShips']:
            if ship.halite + state['currentHalite'] >= 500:
                possibleShips.append(ship)
        closest = closest_thing(Point(tx, ty),possibleShips)
        if closest != None:
            action[closest] = (math.inf, closest, Point(tx, ty))
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    elif v > 500 and v > state['shipValue']:
        targetShipyards.append(state['board'].cells[Point(tx, ty)])
        state['currentHalite'] -= 500
    

    state['closestShipyard'] = closest_shipyard(targetShipyards)

def build_farm():
    global farms
    for cell in state['board'].cells.values():
        if dist(cell.position,state['closestShipyard'][cell.position.x][cell.position.y]) == 1:
            if cell.position in farms:
                continue
            farms.append(cell.position)

def control_farm():
    global farms
    for i,farm in enumerate(farms[:]):
        if dist(farm,state['closestShipyard'][farm.x][farm.y]) > 1:
            # Not worth it
            farms.remove(farm)

def spawn():
    # Ship value: 
    '''
    if state['shipValue'] >= 500: 
        return True
    else:
        return False
    '''
    
    # 抄袭
    bank = state['currentHalite']
    haliteMean = state['haliteMean']
    step = state['board'].step
    shipCnt = len(state['myShips'])
    totalShipCnt = len(state['ships'])
    #isBlocked = state['next'][shipyard.cell.position.x][shipyard.cell.position.y]
    isBlocked = 0 #In theory never blocked, as already checked

    if shipCnt >= 60 or step > 330:
        return False

    inArr = (np.array([bank, totalShipCnt, shipCnt, step, haliteMean, isBlocked]) - spawnMean) / spawnStd
    res = W1 @ inArr + b1
    res = np.maximum(res, 0)
    res = W2 @ res + b2
    res = np.maximum(res, 0)
    res = W3 @ res + b3
    #print(res)
    if res > 0:
        return True
    else:
        return False
    

def spawn_tasks():
    shipyards = state['board'].current_player.shipyards
    shipyards.sort(reverse=True, key=lambda shipyard: state['haliteSpread'][shipyard.position.x][shipyard.position.y])
    shouldSpawn = spawn()

    for shipyard in shipyards:
        if state['currentHalite'] >= 500 and not state['next'][shipyard.cell.position.x][shipyard.cell.position.y]:
            if shouldSpawn:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShips']) < 1 and shipyard == shipyards[0]:
                shipyard.next_action = ShipyardAction.SPAWN
                state['currentHalite'] -= 500
            elif len(state['myShipyards']) == 1:
                for pos in get_adjacent(shipyard.position):
                    cell = state['board'].cells[pos]
                    if cell.ship != None and cell.ship.player_id != state['me']:
                        shipyard.next_action = ShipyardAction.SPAWN
                        state['currentHalite'] -= 500
                        return



spawnMean = np.array([4.9859e+03, 6.0502e+01, 2.5001e+01, 1.9415e+02, 2.8910e+01, 6.1503e-01])
spawnStd = np.array([8.5868e+03, 1.5326e+01, 1.0737e+01, 1.1549e+02, 1.1789e+01, 4.8660e-01])

W1 = np.array([[-1.5224804e+00,2.4725301E-03,-8.7220293e-01,-1.0598649e+00,
   9.9166840e-01,1.8315561e+00],
 [-4.8011017e-01,-6.7499268e-01 ,3.5633636e-01,-1.7301080e+00,
   2.0809724e+00,-8.9656311e-01],
 [-1.1370039e+00,-2.0581658e-01,-2.6484251e+00,-1.5524467e+00,
   3.5835698e+00,-1.7890360e+00],
 [-1.7479208e-01 ,1.9892944e-01, 1.4682317e-01 , 1.1079860e+00,
   1.4466201e-01 , 1.9152831e+00]])
b1 = np.array([1.177493, 0.5530099, 0.1025302, 2.165062 ])

W2 = np.array([[ 0.22407304 ,-0.32596582 ,-0.31062314 ,-0.17025752],
 [-3.6107817 ,  1.9571906 , -0.04028177, -4.0320687 ],
 [ 4.130036  , -1.2309656,  -0.52751654,  1.5594524 ],
 [-0.33959138, -0.0332855 , -0.26249635, -0.35909724]])
b2 = np.array([-0.40560475 ,-0.00167005 , 0.7714385 , -0.19049597])

W3 = np.array([[ 0.4247551 ,  5.073255   ,-4.3405128  , 0.00574893]])
b3 = np.array([-0.2889765])

# General calculations whose values are expected to be used in multiple instances
# Basically calc in botv1.0. 
# Run in update() - see dependency.py

def encode():
    global state
    
    N = state['configuration'].size

    # Halite 
    state['haliteMap'] = np.zeros((N, N))
    for cell in state['cells']:
        state['haliteMap'][cell.position.x][cell.position.y] = cell.halite
    # Halite Spread
    state['haliteSpread'] = np.copy(state['haliteMap'])
    for i in range(1,5):
        state['haliteSpread'] += np.roll(state['haliteMap'],i,axis=0) * 0.5**i
        state['haliteSpread'] += np.roll(state['haliteMap'],-i,axis=0) * 0.5**i
    temp = state['haliteSpread'].copy()
    for i in range(1,5):
        state['haliteSpread'] += np.roll(temp,i,axis=1) * 0.5**i
        state['haliteSpread'] += np.roll(temp,-i,axis=1) * 0.5**i
    # Ships
    state['shipMap'] = np.zeros((state['playerNum'], N, N))
    state['enemyShips'] = []
    for ship in state['ships']:
        state['shipMap'][ship.player_id][ship.position.x][ship.position.y] = 1
        if ship.player_id != state['me']:
            state['enemyShips'].append(ship)
    # Shipyards
    state['shipyardMap'] = np.zeros((state['playerNum'], N, N))
    state['enemyShipyards'] = []
    for shipyard in state['shipyards']:
        state['shipyardMap'][shipyard.player_id][shipyard.position.x][shipyard.position.y] = 1
        if shipyard.player_id != state['me']:
            state['enemyShipyards'].append(shipyard)
    # Total Halite
    state['haliteTotal'] = np.sum(state['haliteMap'])
    # Mean Halite 
    state['haliteMean'] = state['haliteTotal'] / (N**2)
    # Estimated "value" of a ship
    #totalShips = len(state['ships'])
    #state['shipValue'] = state['haliteTotal'] / state
    state['shipValue'] = ship_value()
    # Friendly units
    state['ally'] = state['shipMap'][state['me']]
    # Friendly shipyards
    state['allyShipyard'] = state['shipyardMap'][state['me']]
    # Enemy units
    state['enemy'] = np.sum(state['shipMap'], axis=0) - state['ally']
    # Enemy shipyards
    state['enemyShipyard'] = np.sum(state['shipyardMap'], axis=0) - state['allyShipyard']
    # Closest shipyard
    state['closestShipyard'] = closest_shipyard(state['myShipyards'])
    # Control map
    state['controlMap'] = control_map(state['ally']-state['enemy'],state['allyShipyard']-state['enemyShipyard'])
    state['negativeControlMap'] = control_map(-state['enemy'],-state['enemyShipyard'])
    state['positiveControlMap'] = control_map(state['ally'],state['allyShipyard'])
    # Enemy ship labeled by halite. If none, infinity
    state['enemyShipHalite'] = np.zeros((N, N))
    state['shipHalite'] = np.zeros((state['playerNum'], N, N))
    state['shipHalite'] += np.Infinity
    state['enemyShipHalite'] += np.Infinity
    for ship in state['ships']:
        state['shipHalite'][ship.player.id][ship.position.x][ship.position.y] = ship.halite
        if ship.player.id != state['me']:
            state['enemyShipHalite'][ship.position.x][ship.position.y] = ship.halite
    # Immediate danger map
    state['trapped'] = np.zeros((state['playerNum'], N, N))
    for player in range(state['playerNum']):
        state['trapped'][player] = get_immediate_danger(player)
    # Avoidance map (Places not to go for each ship)
    for ship in state['myShips']:
        state[ship] = {}
        state[ship]['blocked'] = get_avoidance(ship)
        state[ship]['danger'] = get_danger(ship.halite)
    state['generalDangerMap'] = get_danger(1)
    # Who we should attack
    if len(state['board'].opponents) > 0:
        state['killTarget'] = get_target()
    
def get_avoidance(s):
    threshold = s.halite
    #Enemy units
    temp = np.where(state['enemyShipHalite'] < threshold, 1, 0)
    enemyBlock = np.copy(temp)
    enemyBlock = enemyBlock + np.roll(temp,1,axis=0)
    enemyBlock = enemyBlock + np.roll(temp,-1,axis=0)
    enemyBlock = enemyBlock + np.roll(temp,1,axis=1)
    enemyBlock = enemyBlock + np.roll(temp,-1,axis=1)

    enemyBlock = enemyBlock + state['enemyShipyard']

    blocked = enemyBlock
    blocked = np.where(blocked>0,1,0)
    return blocked

def get_danger(s):
    threshold = s
    dangerMap = np.where(state['enemyShipHalite'] < threshold, 1, 0)
    temp = dangerMap.copy()
    for i in range(1,4):
        dangerMap = np.add(dangerMap,np.roll(temp,i,axis=0) * 0.7**i,casting="unsafe")
        dangerMap += np.roll(temp,-i,axis=0) * 0.7**i
    temp = dangerMap.copy()
    for i in range(1,4):
        dangerMap += np.roll(temp,i,axis=1) * 0.7**i
        dangerMap += np.roll(temp,-i,axis=1) * 0.7**i
    return dangerMap
    
def closest_shipyard(shipyards):
    N = state['configuration'].size
    res = [[None for y in range(N)]for x in range(N)]
    for x in range(N):
        for y in range(N):
            minimum = math.inf
            for shipyard in shipyards:
                if dist(Point(x,y),shipyard.position) < minimum:
                    minimum = dist(Point(x,y),shipyard.position)
                    res[x][y] = shipyard.position
    return res
    
def control_map(ships,shipyards):
        ITERATIONS = 3

        res = np.copy(ships)
        for i in range(1,ITERATIONS+1):
            res += np.roll(ships,i,axis=0) * 0.5**i
            res += np.roll(ships,-i,axis=0) * 0.5**i
        temp = res.copy()
        for i in range(1,ITERATIONS+1):
            res += np.roll(temp,i,axis=1) * 0.5**i
            res += np.roll(temp,-i,axis=1) * 0.5**i
        
        return res + shipyards
        
def get_target():
    board = state['board']
    me = board.current_player
    idx,v = 0, -math.inf
    for i,opponent in enumerate(board.opponents):
        value = 0
        if opponent.halite-me.halite > 0:
            value = -(opponent.halite-me.halite)
        else:
            value = (opponent.halite-me.halite) * 5
        if value > v:
            v = value
            idx = i
    return board.opponents[idx]

def get_immediate_danger(team):
    res = np.zeros((state['configuration'].size,state['configuration'].size))
    enemy = np.zeros((state['configuration'].size,state['configuration'].size))
    for i in range(state['playerNum']):
        if i == team:
            continue
        enemy += np.where(state['shipHalite'][i]==0,1,0)
    for axis in range(2):
        secondAxis = 0 if axis == 1 else 1
        for direction in [-1,1]:
            N = enemy.copy()
            N += np.roll(enemy,direction,axis=axis)
            N += np.roll(np.roll(enemy,direction,axis=axis),1,axis=secondAxis)
            N += np.roll(np.roll(enemy,direction,axis=axis),-1,axis=secondAxis)
            N += np.roll(N,direction,axis=axis)
            N += np.roll(N,direction,axis=axis)
            '''N += np.roll(np.roll(enemy,direction*3,axis=axis),2,axis=secondAxis)
            N += np.roll(np.roll(enemy,direction*3,axis=axis),-2,axis=secondAxis)'''
            res += np.where(N>0,1,0)
    danger = np.where(res>=4,1,0)
    return danger
            

        
# Direction from point s to point t
def direction_to(s: Point, t: Point) -> ShipAction:
    candidate = directions_to(s, t)
    if len(candidate) == 2:
        if dist(Point(s.x,0),point(t.x,0)) > dist(Point(0,s.y),Point(0,t.y)):
            return candidate[1]
        else:
            return candidate[0]
    elif len(candidate) == 1:
        random.choice(candidate)
    else:
        return None

# Distance from point a to b
def dist(a: Point, b: Point) -> int:
    N = state['configuration'].size
    return min(abs(a.x - b.x), N - abs(a.x - b.x)) + min(abs(a.y - b.y), N - abs(a.y - b.y))

# Returns list of possible directions
def directions_to(s: Point, t: Point) -> ShipAction:
    N = state['configuration'].size
    candidates = [] # [N/S, E/W]
    if s.x-t.x != 0:
        candidates.append(ShipAction.WEST if (s.x-t.x) % N < (t.x-s.x) % N else ShipAction.EAST)
    if s.y-t.y != 0:
        candidates.append(ShipAction.SOUTH if (s.y-t.y) % N < (t.y-s.y) % N else ShipAction.NORTH)
    return candidates

# Deserialize an integer which represents a point
def unpack(n) -> Point:
    N = state['configuration'].size
    return Point(n // N, n % N)

# A default direction to target
def direction_to(s: Point, t: Point) -> ShipAction:
    candidate = directions_to(s, t)
    return random.choice(candidate) if len(candidate) > 0 else None

# Returns the "next" point of a ship at point s with shipAction d
def dry_move(s: Point, d: ShipAction) -> Point:
    N = state['configuration'].size
    if d == ShipAction.NORTH:
        return s.translate(Point(0, 1),N)
    elif d == ShipAction.SOUTH:
        return s.translate(Point(0, -1),N)
    elif d == ShipAction.EAST:
        return s.translate(Point(1, 0),N)
    elif d == ShipAction.WEST:
        return s.translate(Point(-1, 0),N)
    else:
        return s
    
# Returns opposite direction
def opp_direction(d: ShipAction):
    if d == ShipAction.NORTH:
        return ShipAction.SOUTH
    if d == ShipAction.SOUTH:
        return ShipAction.NORTH
    if d == ShipAction.WEST:
        return ShipAction.EAST
    if d == ShipAction.EAST:
        return ShipAction.WEST
    return None

# Returns list of len 4 of adjacent points to a point
def get_adjacent(point):
    N = state['configuration'].size
    res = []
    for offX, offY in ((0,1),(1,0),(0,-1),(-1,0)):
        res.append(point.translate(Point(offX,offY),N))
    return res
    
def safe_naive(s,t,blocked):
    for direction in directions_to(s.position,t):
        target = dry_move(s.position,direction)
        if not blocked[target.x][target.y]:
            return direction
    return None

def move_cost(s : Ship, t : Point, p : Point):
    navigationWeights = weights[6]
    cost = state[s]['danger'][p.x][p.y] * navigationWeights[1]
    c = state['board'].cells[p]
    if c.ship != None and c.ship.player_id != state['me']:
        if direction_to(t,s.position) != direction_to(t,p):
            cost += 1
    
    if s.halite > 0 and state['trapped'][state['me']][s.position.x][s.position.y]:
        cost += 5
    
    return cost

# Dijkstra's movement
def d_move(s : Ship, t : Point, inBlocked):

    nextMap = state['next']
    sPos = s.position
    blocked = inBlocked + nextMap
    # Check if we are trying to attack
    if state['board'].cells[t].ship != None:
        target = state['board'].cells[t].ship
        if target.player_id != state['me'] and target.halite == s.halite:
            blocked[t.x][t.y] -= 1
    elif state['board'].cells[t].shipyard != None and state['board'].cells[t].shipyard.player_id != state['me']:
        blocked[t.x][t.y] -= 1
    # Don't ram stuff thats not the target.
    if state['board'].step < state['configuration']['episodeSteps'] - state['configuration'].size * 1.5:
        blocked += np.where(state['enemyShipHalite'] <= s.halite,1,0)
        temp = np.zeros(blocked.shape)
        tot = 0
        
        for pos in get_adjacent(sPos):
            if state['allyShipyard'][pos.x][pos.y]:
                continue
            if blocked[pos.x][pos.y] > 0:
                tot += 1
            else:
                for tPos in get_adjacent(pos):
                    if state['enemyShipHalite'][tPos.x][tPos.y] <= s.halite:
                        if tPos == t:
                            continue
                        tot += 1
                        temp[pos.x][pos.y] = 1
                        break
        
        if not(tot == 4 and (state['board'].cells[sPos].halite > 0 or nextMap[sPos.x][sPos.y])):
            blocked += temp
            
    blocked = np.where(blocked>0,1,0)

    desired = None

    #Stay still
    if sPos == t or nextMap[t.x][t.y]:

        #Someone with higher priority needs position, must move. Or being attacked.
        if blocked[t.x][t.y]:
            for processPoint in get_adjacent(sPos):
                if not blocked[processPoint.x][processPoint.y]:
                    #nextMap[processPoint.x][processPoint.y] = 1
                    desired = direction_to(sPos,processPoint)
                    t = processPoint
            if desired == None:
                target = micro_run(s)
                t = dry_move(sPos,target)
                desired = target
        else:
            t = sPos
            desired = None
    else:
        #Dijkstra
        pred = {}
        calcDist = {}
        pq = PriorityQueue()
        pqMap = {}

        pqMap[dist(sPos,t)] = [sPos]
        pq.put(dist(sPos,t))
        pred[sPos] = sPos
        calcDist[sPos] = dist(sPos,t)

            # Main

        while not pq.empty():
            if t in calcDist:
                break
            currentPoint = pqMap.get(pq.get()).pop()
            for processPoint in get_adjacent(currentPoint):
                if blocked[processPoint.x][processPoint.y] or processPoint in calcDist: 
                    continue
                calcDist[processPoint] = calcDist[currentPoint] + 1 + move_cost(s,t,processPoint)
                priority = calcDist[processPoint]
                pqMap[priority] = pqMap.get(priority,[])
                pqMap[priority].append(processPoint)
                pq.put(priority)
                pred[processPoint] = currentPoint

        if not t in pred:

            # Can go in general direction
            res = safe_naive(s,t,blocked)
            if res != None:
                t = dry_move(s.position,res)
                desired = res
            else:
                #Random move
                for processPoint in get_adjacent(sPos):
                    if not blocked[processPoint.x][processPoint.y]:
                        #nextMap[processPoint.x][processPoint.y] = 1
                        t = processPoint
                        desired = direction_to(sPos,processPoint)
                
                # Run
                if desired == None and blocked[sPos.x][sPos.y]:
                    target = micro_run(s)
                    t = dry_move(sPos,target)
                    desired = target
                elif not blocked[sPos.x][sPos.y]:
                    t = sPos
                    desired = None        
        else:
            # Path reconstruction
            while pred[t] != sPos:
                t = pred[t]

            desired = direction_to(sPos,t)

    # Reduce collisions
    if desired != None and state['board'].cells[t].ship != None and state['board'].cells[t].ship.player_id == state['me']:
        target = state['board'].cells[t].ship
        s.next_action = desired
        if action[target] != True:
            nextMap[t.x][t.y] = 1
            result = process_action(action[target])
            # Going there will kill it
            if result == None or result == ShipAction.CONVERT:
                desired = d_move(s,t,inBlocked)
                t = dry_move(sPos,desired)
    nextMap[t.x][t.y] = 1
    return desired

# Ship might die, RUN!
def micro_run(s):
    sPos = s.position
    nextMap = state['next']

    if state[s]['blocked'][sPos.x][sPos.y]:
        if s.halite > 400:
            return ShipAction.CONVERT
        score = [0,0,0,0]

        # Preprocess
        directAttackers = 0
        for i,pos in enumerate(get_adjacent(sPos)):
            if state['enemyShipHalite'][pos.x][pos.y] < s.halite:
                directAttackers += 1

        # Calculate score
        for i,pos in enumerate(get_adjacent(sPos)):
            score[i] = 0
            for j,tPos in enumerate(get_adjacent(sPos)):
                if state['enemyShipHalite'][tPos.x][tPos.y] < s.halite:
                    score[i] -= 0.5
            if state['enemyShipHalite'][pos.x][pos.y] < s.halite:
                score[i] -= 0.5 + 1/directAttackers
            score[i] += state['negativeControlMap'][pos.x][pos.y] * 0.01
        # Select best position
        i, maximum = 0,0 
        for j, thing in enumerate(score):
            if thing > maximum:
                i = j
                maximum = thing
        return direction_to(sPos,get_adjacent(sPos)[i])
    else:
        return None






# Key function
# For a ship, return the inherent "value" of the ship to get to a target cell

def get_reward(ship,target):
    
    cell = target[0]
    res = 0
    # Don't be stupid
    if state[ship]['blocked'][cell.position.x][cell.position.y] and cell.shipyard == None:
        res = 0
    elif target[1] == 'cell':
        # Mining reward
        if (cell.ship is None or cell.ship.player_id == state['me']) and cell.halite > 0:
            res = mine_reward(ship,cell)
        elif cell.shipyard is None and cell.halite == 0 and (cell.ship is None or cell.ship.player_id == state['me']):
            res = control_reward(ship,cell)
        elif cell.ship is not None and cell.ship.player_id != state['me']:
            res = attack_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id == state['me']:
            res = return_reward(ship,cell)
        elif cell.shipyard is not None and cell.shipyard.player_id != state['me']:
            res = attack_reward(ship,cell)
    elif target[1] == 'guard':
        res = guard_reward(ship,cell)
    return res

def control_reward(ship,cell):

    return 0
    
    sPos = ship.position
    cPos = cell.position

    if ship.halite > 0 or dist(cPos,state['closestShipyard'][cPos.x][cPos.y]) <= 2:
        return 0
    res = 0
    for pos in get_adjacent(cPos):
        tCell = state['board'].cells[pos]
        if tCell.halite > 0:
            res += 3.5
    res -= dist(sPos,cPos) + dist(cPos,state['closestShipyard'][cPos.x][cPos.y])
    return res

def guard_reward(ship,cell):
    cPos = cell.position
    sPos = ship.position
    guardWeights = weights[5]
    if len(state['enemyShips']) == 0:
        return 0
    closestEnemy = closest_thing(ship.position,state['enemyShips'])
    if dist(sPos,cPos) > dist(closestEnemy.position,cPos):
        return 0
    elif ship.halite != 0 and dist(sPos,cPos) >= dist(closestEnemy.position,cPos):
        return 0

    # Check if we want to build
    if cell.shipyard == max(state['myShipyards'],key=lambda shipyard: state['haliteSpread'][shipyard.position.x][shipyard.position.y]):
        if state['currentHalite'] >= 500 and state['spawn']:
            return 0
    
    return guardWeights[0] / (dist(closestEnemy.position,cPos) * max(dist(sPos,cPos),1))
 
def mine_reward(ship,cell):

    mineWeights = weights[1]
    sPos = ship.position
    cPos = cell.position
    cHalite = cell.halite
    cell
    shipyardDist = dist(cPos,state['closestShipyard'][cPos.x][cPos.y])

    if state['generalDangerMap'][cPos.x][cPos.y] > 1.5 and state['trapped'][state['me']][cPos.x][cPos.y]:
        return 0

    # Halite per turn
    halitePerTurn = 0

    # Occupied cell
    if cell.ship != None and cell.ship.player_id == state['me'] and cell.ship.halite <= ship.halite:
        # Current cell multiplier
        if sPos == cPos:
            if cHalite > state['haliteMean'] * mineWeights[2] and cHalite > 10 and ship.halite > 0:
                cHalite = cHalite * mineWeights[1]

        # Farming!
        if cPos in farms and cell.halite < min(500,(state['board'].step + 10*15)) and state['board'].step < state['configuration']['episodeSteps'] - 50:
            return 0
        
        if shipyardDist >= 3:
            # Don't mine if enemy near
            for pos in get_adjacent(cPos):
                if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                    return 0
            
            if state['trapped'][state['me']][cPos.x][cPos.y]:
                return 0
    
    # Dangerous area
    cHalite += state['negativeControlMap'][cPos.x][cPos.y] * mineWeights[4]
    
    if state['enemyShipHalite'][cPos.x][cPos.y] <= ship.halite:
        return 0
    for pos in get_adjacent(cPos):
        if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
            return 0
        
    '''
    if state['currentHalite'] > 1000: # Do we need some funds to do stuff?
        # No
        halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),0) 
    else:
        # Yes
        halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),dist(cPos,state['closestShipyard'][cPos.x][cPos.y]))
    '''
    halitePerTurn = halite_per_turn(cHalite,dist(sPos,cPos),shipyardDist) 
    # Surrounding halite
    spreadGain = state['haliteSpread'][cPos.x][cPos.y] * mineWeights[0]
    res = halitePerTurn + spreadGain

    if state[ship]['danger'][cPos.x][cPos.y] > 1.3:
        res -= mineWeights[3] ** state[ship]['danger'][cPos.x][cPos.y]
        
    return res

def attack_reward(ship,cell):

    attackWeights = weights[2]
    cPos = cell.position 
    sPos = ship.position
    d = dist(ship.position,cell.position)
    
    # Don't even bother
    if dist(sPos,cPos) > 6:
        return 0

    res = 0
    # It's a ship!
    if cell.ship != None:
            # Nearby 
        if cPos in get_adjacent(sPos) and state['controlMap'][cPos.x][cPos.y] < 0.5:
            # Try to reduce collision num
            for pos in get_adjacent(cPos):
                if state['enemyShipHalite'][pos.x][pos.y] <= ship.halite:
                    return 0

        if cell.ship.halite > ship.halite:
            # Defend the farm!
            if cPos in farms:
                return cell.halite - d
            res = max([cell.halite**(attackWeights[4]),state['controlMap'][cPos.x][cPos.y]*attackWeights[2]]) - d*attackWeights[3]
        elif len(state['myShips']) > 15:
            res = state['controlMap'][cPos.x][cPos.y] * 100 / d**2
        if ship.halite != 0:
            res = res / 3
    
    # It's a shipyard!
    elif len(state['myShips']) > 10 and ship.halite == 0:
        if len(state['myShips']) > 15 and cell.shipyard.player == state['killTarget']:
            # Is it viable to attack
            viable = True
            for pos in get_adjacent(cPos):
                target = state['board'].cells[pos].ship
                if target != None and target.player_id != state['me'] and target.halite <= ship.halite:
                    viable = False
                    break
            if viable:
                res = attackWeights[1] / d**2
        
        res = max(res,state['controlMap'][cPos.x][cPos.y] * 100 / d**2)

    return res * attackWeights[0]

def return_reward(ship,cell):

    returnWeights = weights[3]
    sPos = ship.position
    cPos = cell.position

    if sPos == cPos :
        return 0
    res = 0
    
    if state['currentHalite'] > 1000:
        res = ship.halite / (dist(sPos,cPos)) * returnWeights[0]
    else:
        res = ship.halite / (dist(sPos,cPos))
    
    res = res * returnWeights[1]
    return res 

def shipyard_value(cell):
    # Features
    shipyardWeights = weights[0]
    cPos = cell.position

    if state['board'].step > 310:
        return 0

    nearestShipyard = closest_thing(cPos,state['shipyards'])
    nearestShipyardDistance = 1
    if nearestShipyard != None:
        nearestShipyardDistance = dist(nearestShipyard.position,cPos)
    negativeControl = min(0,state['controlMap'][cPos.x][cPos.y])
    if len(state['myShips']) > 0:
        negativeControl = max(negativeControl-0.5 ** dist(closest_thing(cPos,state['myShips']).position,cPos),state['negativeControlMap'][cPos.x][cPos.y])
    haliteSpread = state['haliteSpread'][cPos.x][cPos.y] - state['haliteMap'][cPos.x][cPos.y]
    shipShipyardRatio = len(state['myShips']) / max(1,len(state['myShipyards']))

    # Hard limit on range and halite spread
    if nearestShipyardDistance <= 5 or haliteSpread <= 200:
        return 0

    # Base halite multiplier
    res = haliteSpread * shipyardWeights[0]

    # Negative control
    res += negativeControl * shipyardWeights[1]

    # Nearest shipyard
    res = res * nearestShipyardDistance ** shipyardWeights[2]

    # Ship shipyard ratio multiplier
    res = res * shipShipyardRatio ** shipyardWeights[3]

    # Final multiplier and bias
    res = res * shipyardWeights[4] + shipyardWeights[5]

    return res

def ship_value():
    if len(state['myShips']) >= 60:
        return 0
    res = state['haliteMean'] * 0.25 * (state['configuration']['episodeSteps']- 30 - state['board'].step) * weights[4][0]
    res += (len(state['ships']) - len(state['myShips'])) ** 1.5 * weights[4][1]
    res += len(state['myShips'])  ** 1.5 * weights[4][2]
    return res 
        



        

# The final function


@board_agent
def agent(board):

    print("Turn =",board.step+1)
    # Init
    if board.step == 0:
        init(board)

    # Update
    update(board)

    # Convert
    convert_tasks()

    # Farm
    #farm_tasks()

    # Ship
    ship_tasks()

    # Spawn
    spawn_tasks()
