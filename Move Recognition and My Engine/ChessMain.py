'''
This is our main driver file . It will be responsible for handling user input and displaying the current GameState object.
'''
import pygame as p
import ChessEngine
import SmartMoveFinder
import OpeningBook as ob
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import imutils
import time
import pyttsx3

engine = pyttsx3.init()
# engine.setProperty('rate', 70)

'''
Defining global variables
'''
BOARD_WIDTH = BOARD_HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 300
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT
DIMENSION = 8 # dimensions of a chess board are 8x8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
MAX_FPS = 15 # will come in play for animations later on 
IMAGES = {} 

'''
Initializing a global dictionary of Images. This will be called exactly once in the main.
'''
def load_Images():
    pieces = ["bR", "bN", "bB", "bQ", "bK", "bp", "wp", "wR", "wN", "wB", "wQ", "wK"]

    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("Images/" + piece + ".png"), (SQ_SIZE,SQ_SIZE))

'''
drawGameState is resposible for all the graphics within a current game state.
'''

def drawGameState(screen, gs, validMoves, moveLogFont):
    drawBoard(screen)  # This function will draw the squares on the board 
    drawPieces(screen, gs.board)  # This will draw pieces on top of that board.
    drawMoveLog(screen, gs, moveLogFont)

'''
Draw the squares on the board. (235, 235, 208) and (119, 148, 85) are the chess.com colours.
'''
def drawBoard(screen):
    global colors
    colors = [(235, 235, 208), (119, 148, 85)]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r+c) % 2)]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

'''
Draw the pieces on the board using the current GameSate,board .
'''

def drawPieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

'''
Draws the move log
'''
def drawMoveLog(screen , gs, font):
    pass
    moveLogRect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, (0, 0, 0), moveLogRect)
    moveLog = gs.moveLog
    moveTexts = []
    for i in range(0, len(moveLog), 2) :
        moveString = str(i//2 + 1) + "." + moveLog[i].getChessNotation() + " "
        if i + 1 < len(moveLog) :    # Just to make sure that black moved
            moveString += moveLog[i + 1].getChessNotation()
        moveTexts.append(moveString) 
    movesPerRow = 2 
    padding = 5
    textY = padding
    linespacing  = 2
    for i in range(0, len(moveTexts), movesPerRow):
        text = ""
        for j in range(movesPerRow):
            if i + j < len(moveTexts) :
                text += moveTexts[i + j] + "       "
        textObject = font.render(text, True, (255, 255, 255))
        testLocation = moveLogRect.move(padding, textY)
        screen.blit(textObject,testLocation)
        textY += textObject.get_height() + linespacing

'''
Animating a move
'''    
def animateMove(move, screen, board, clock):            
    global colors
    dR = move.endRow - move.startRow
    dC = move.endCol - move.startCol
    framesPerSquare = 3 # Frames to move one square
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.startRow + dR*frame/frameCount, move.startCol + dC*frame/frameCount)
        drawBoard(screen)
        drawPieces(screen, board)
        # Erase the piece moved from its ending square
        color = colors[(move.endRow + move.endCol) % 2]
        endSquare = p.Rect(move.endCol*SQ_SIZE, move.endRow*SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, endSquare)
        # Draw the Captured piece onto rectangle
        if move.pieceCaptured != "--" :
            if move.isEnpassantMove :
                enPassantRow = move.endRow + 1 if move.pieceCaptured[0] == "b" else move.endRow - 1
                endSquare = p.Rect(move.endCol*SQ_SIZE, enPassantRow*SQ_SIZE, SQ_SIZE, SQ_SIZE)

            screen.blit(IMAGES[move.pieceCaptured], endSquare)
        # Draw the moving piece
        screen.blit(IMAGES[move.pieceMoved], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(60)    

def drawEndGameText(screen, text):
    font = p.font.SysFont("Helvitca", 32, True, False)
    textObject = font.render(text, 0, (0, 0, 0))
    textLocation = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(BOARD_WIDTH/2 - textObject.get_width()/2, BOARD_HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject,textLocation)

def findSquareContours(CalibratingImg):
    chessBoardOriginal = cv2.imread(CalibratingImg)

    chessBoard = chessBoardOriginal.copy()
    chessBoard = cv2.cvtColor(chessBoard, cv2.COLOR_BGR2GRAY)
    chessBoard = cv2.Canny(chessBoard, 50, 150)

    cdst = cv2.cvtColor(chessBoard, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv2.HoughLines(chessBoard, 1, np.pi / 180, 120, None, 0, 0)

    img = np.zeros(chessBoard.shape, dtype=np.uint8)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    req_cntrs = []

    count = 0
    index = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 900 and area < 3600:
            req_cntrs.append(cnt)
            cv2.drawContours(chessBoardOriginal, contours,
                             index, (0, 255, 0), 2)
            count += 1
        index += 1
    cv2.imshow("Contours detected", chessBoardOriginal)
    # time.sleep(10)

    return req_cntrs

def ImageCapture():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite(filename='Images/CalibratingImage.jpg', img=frame)
                webcam.release()
                cv2.destroyAllWindows()
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

def CroppingImage(image):

    img = cv2.imread(image)

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([179, 255, 168])
    mask = cv2.inRange(imgHSV, lower, upper)
    mask = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mx = (0, 0, 0, 0)
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx

    roi = img[y + 15:y+h - 15, x + 15:x+w - 15]
    cv2.imwrite("Images/Image_crop.jpg", roi)

    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 2)
    cv2.imwrite('Images/Image_cont.jpg', img)

    return x, y, w, h

def cntrs2squares():

    contours = findSquareContours("Images/Image_crop.jpg")
    img = cv2.imread("Images/Image_crop.jpg")
    cv2.drawContours(img, contours, -1, (255, 255, 255))
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    
    def contoursortX(c):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        return cX

    def contoursortY(c):
        M = cv2.moments(c)
        cY = int(M["m01"] / M["m00"])
        return -cY

    number_to_alpha = {0 :"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}

    contours.sort(key= contoursortX)

    new_contours = []

    for i in range(8):
        alpha =  number_to_alpha[i]
        new_list = contours[8*i: 8*i + 8]
        new_list.sort(key = contoursortY)
        for i in range(8):
            new_contours.append([new_list[i], alpha + str(i + 1)])

    return new_contours

def pieceMovement():
    imageA = cv2.imread("LOL.jpg")
    imageB = cv2.imread("LMAO.jpg")

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts1, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for cnt in cnts1:
    	area = cv2.contourArea(cnt)
    	if area >= 200 and area <= 3600 :
    		cnts.append(cnt)

    # cv2.imshow("Diff", diff)
    # cv2.imshow("Thresh", thresh)
    # cv2.waitKey(0)

    cnts.sort (key = lambda x: cv2.contourArea (x), reverse = True)
    max_cnt = cnts[0]
    print("Area of the contour with the max Area : " + str(cv2.contourArea(max_cnt)))
    x, y, w, h = cv2.boundingRect(max_cnt)

    max_cnt2 = cnts[1]
    a, b, c, d = cv2.boundingRect(max_cnt2)
    print("Area of the contour with the second largest Area : " + str(cv2.contourArea(max_cnt2)))
    return(x, y, w, h), (a, b, c, d)

def setBoard(x, y, w, h):
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        check, chessBoardOriginal = webcam.read()
        chessBoardOriginal = chessBoardOriginal[y + 15:y+h - 15, x + 15:x+w - 15]

        cv2.imshow("Live Feed", chessBoardOriginal)

        key = cv2. waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='LOL.jpg', img=chessBoardOriginal)
            cv2.imwrite(filename='LMAO.jpg', img=chessBoardOriginal)
            webcam.release()
            cv2.destroyAllWindows()
            break

def firstMove(x, y, w, h):
    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        check, chessBoardOriginal = webcam.read()
        chessBoardOriginal = chessBoardOriginal[y + 15:y+h - 15, x + 15:x+w - 15]

        cv2.imshow("Live Feed", chessBoardOriginal)

        key = cv2. waitKey(1)
        cv2.imwrite(filename='LMAO.jpg', img=chessBoardOriginal)
        webcam.release()
        cv2.destroyAllWindows()
        break

def getMoveSquares(dimensions, whiteToMove, board):

    webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    x, y, w, h = dimensions
    while True:
        check, chessBoardOriginal = webcam.read()
        chessBoardOriginal = chessBoardOriginal[y + 15:y+h - 15, x + 15:x+w - 15]

        cv2.imwrite(filename='LMAO.jpg', img=chessBoardOriginal)
        webcam.release()
        cv2.destroyAllWindows()
        break

    end, start = pieceMovement()
    p, q, r, s = end
    a, b, c, d = start
    endPnt = p+r//2, q+s//2
    startPnt = a+c//2, b+d//2

    imageA = cv2.imread("LOL.jpg")
    imageB = cv2.imread("LMAO.jpg")

    cv2.rectangle(imageB, (p, q), (p + r, q + s), (0, 0, 255), 2)
    cv2.rectangle(imageA, (a, b), (a + c, b + d), (0, 0, 255), 2)

    # cv2.imshow("Original", imageA)
    # cv2.imshow("Modified", imageB)

    # cv2.waitKey(0)

    startSq = "a1"
    endSq = "h8"

    for item in finalList:
        cnt = item[0]
        if cv2.pointPolygonTest(cnt, endPnt, False) == 1.0 :
            endSq = item[1]
        if cv2.pointPolygonTest(cnt, startPnt, False) == 1.0 :
            startSq = item[1]

    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,"5": 3, "6": 2, "7": 1, "8": 0}
    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}

    startSqcol = filesToCols[startSq[0]]
    startSqrow = ranksToRows[startSq[1]]

    endSqcol = filesToCols[endSq[0]]
    endSqrow = ranksToRows[endSq[1]]

    if whiteToMove :
        if board[endSqrow][endSqcol][0] != "w":
            endSqcolf, endSqrowf = endSqcol, endSqrow
            startSqcolf, startSqrowf = startSqcol, startSqrow
        else :
            endSqcolf, endSqrowf = startSqcol, startSqrow
            startSqcolf, startSqrowf = endSqcol, endSqrow
    else :
        if board[endSqrow][endSqcol][0] != "b":
            endSqcolf, endSqrowf = endSqcol, endSqrow
            startSqcolf, startSqrowf = startSqcol, startSqrow
        else :
            endSqcolf, endSqrowf = startSqcol, startSqrow
            startSqcolf, startSqrowf = endSqcol, endSqrow

    cv2.imwrite(filename='LOL.jpg', img=chessBoardOriginal)
    return startSqrowf, startSqcolf, endSqrowf, endSqcolf

'''
The main driver of our code. This will handle user in put and updating th graphics
'''           
'''
Initializing Everything
'''
p.init()
screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
clock = p.time.Clock()
screen.fill(p.Color("white"))
gs = ChessEngine.GameState()
load_Images()

print("Capturing Calibrating Image")
ImageCapture()
x, y, w, h = CroppingImage("Images/CalibratingImage.jpg")
finalList = cntrs2squares()
if len(finalList) == 64:
    print("Calibration Successfull !!!")
else :
    print(len(finalList))
    print("Calibration Failed!!!")

print("Set the pieces on the board")

image = cv2.imread("Images/Image_crop.jpg")
contours = []
for item in finalList:
    c = item[0]
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(image, item[1], (cX, cY), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 1)
cv2.imshow("Contours and Squares", image)
cv2.waitKey(0)

setBoard(x, y, w, h)

moveLogFont = p.font.SysFont("Arial", 20, False, False)

validMoves = gs.getValidMoves()
moveMade = False # Flag variable for when a move is made
animate = False # Flag variable for when we should animate a move

running = True

gameOver = False
draw = False

playerOne = True # If a human is playing white , than this will be True . If an AI is playing , then this will be False .
playerTwo = True # This is for Black

pgn = ""

while running:

    humanTurn = (gs.whiteToMove and playerOne) or (not gs.whiteToMove and playerTwo)

    for e in p.event.get():
        if e.type == p.QUIT:
            running = False

        elif e.type == p.KEYDOWN:
            if e.key == p.K_t :
                print("t key pressed")
                if not gameOver and humanTurn :
                    print("Taking a pic of what you have done")
                    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,"5": 3, "6": 2, "7": 1, "8": 0}
                    rowsToRanks = {v: k for k, v in ranksToRows.items()}
                    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
                    colsToFiles = {v: k for k, v in filesToCols.items()} 

                    startrow, startcol, endrow, endcol = getMoveSquares((x, y, w, h), gs.whiteToMove, gs.board)

                    startSq = colsToFiles[startcol] + rowsToRanks[startrow]
                    endSq = colsToFiles[endcol] + rowsToRanks[endrow]
                    MOVE = startSq + endSq

                    move = ChessEngine.Move((startrow, startcol), (endrow, endcol), gs.board)
                    for i in range(len(validMoves)) :
                        if move == validMoves[i] :     
                            gs.makeMove(validMoves[i])
                            for log in gs.boardLog :
                                if gs.boardLog.count(log) > 2 :
                                    draw = True 
                            moveMade = True
                            animate = True 

                    if not moveMade :
                        print(MOVE)
                        print(startrow, startcol, endrow, endcol)

            elif e.key == p.K_z : # Undo when z is pressed        
                    gs.undoMove()
                    moveMade = True
                    animate = False
                    gameOver = False

            elif e.key == p.K_r : # Reset the game when r is pressed 
                gs = ChessEngine.GameState()
                validMoves = gs.getValidMoves()
                # sqSelected = ()
                # playerClicks = []
                moveMade = False
                animate = False
                gameOver = False

            elif e.key ==p.K_a : # Getting the FEN and the PGN og the board 
                print("This is the current FEN : " + gs.boardLog[-1])  
                print("PGN : " + pgn)


        # AI move finder logic
    if not gameOver and not humanTurn :

        opnMove = False 

        for i in range(len(ob.opl)):
            strb = ob.opl[i]
            if pgn in strb:

                if not gs.whiteToMove :
                    nmove = strb.replace(pgn, "").split()[0] 
                else :
                    nmove = strb.replace(pgn, "").split()[1]

                for move in validMoves : 
                    if nmove == move.getChessNotation():
                        gs.makeMove(move) 
                        engine.say("The opponent played " + move.getChessNotation())
                        engine.runAndWait()
                        print(move.getChessNotation())
                        print("Make the opponent's move on the board!")
                        print("LOL")
                        time.sleep(10)
                        print("Remove your hands from the board")
                        time.sleep(5)

                        webcam = cv2.VideoCapture(1)
                        while True:
                            check, chessBoardOriginal = webcam.read()
                            chessBoardOriginal = chessBoardOriginal[y + 15:y+h - 15, x + 15:x+w - 15]

                            cv2.imwrite(filename='LOL.jpg', img=chessBoardOriginal)
                            webcam.release()
                            cv2.destroyAllWindows()
                            break

                        print("Now you can make your move")
                        time.sleep(2)

                        opnMove = True

                        break  

                break

        if not opnMove :
            
            AIMove = SmartMoveFinder.findBestMove(gs, validMoves)

            if AIMove is None :
                print("There are no possible AI Moves.")
                print("CheckMate : " + str(gs.checkMate) + " StaleMate : " + str(gs.staleMate) + " Draw : " + str(draw))
                if not gs.staleMate and not gs.checkMate and not draw :
                    gs.makeMove(SmartMoveFinder.findRandomMove(validMoves))
                    print("This is a Random move " + str(SmartMoveFinder.findRandomMove(validMoves).getChessNotation()))         
            else :
                gs.makeMove(AIMove)
                print("This is " + ("black's" if gs.whiteToMove else "white's") + " AI Move: " + AIMove.getChessNotation())
                print((SmartMoveFinder.scoreBoard(gs))/100)
                for log in gs.boardLog :
                    if gs.boardLog.count(log) > 2 :
                        draw = True

            engine.say("The opponent played " + AIMove.getChessNotation())
            engine.runAndWait()
            print("Make the opponent's move on the board!")
            print("LMAO")
            time.sleep(10)
            print("Remove your hands from the board")
            time.sleep(5)

            webcam = cv2.VideoCapture(1)
            while True:
                check, chessBoardOriginal = webcam.read()
                chessBoardOriginal = chessBoardOriginal[y + 15:y+h - 15, x + 15:x+w - 15]

                cv2.imwrite(filename='LOL.jpg', img=chessBoardOriginal)
                webcam.release()
                cv2.destroyAllWindows()
                break
            print("Now you can make your move")
            time.sleep(2)

        moveMade = True
        animate = True      

    if moveMade :
        if animate:
            animateMove(gs.moveLog[-1], screen, gs.board, clock) 
        pgn = ""
            
        for i in range(0, len(gs.moveLog), 2) :
            pgnString = str(i//2 + 1) + ". " + gs.moveLog[i].getChessNotation() + " "
            if i + 1 < len(gs.moveLog):
                pgnString +=gs.moveLog[i + 1].getChessNotation() + " "
            pgn += pgnString

        validMoves = gs.getValidMoves()
        moveMade = False
        animate = False

    drawGameState(screen, gs, validMoves, moveLogFont) 

    if gs.checkMate or gs.staleMate or draw:
        gameOver = True
        drawEndGameText(screen, text = "Stalemate" if gs.staleMate else "Draw" if draw else "Black wins by checkmate" if gs.whiteToMove else "White wins be checkmate")

    clock.tick(MAX_FPS)       
    p.display.flip()