import random   

pieceScore = {"K": 0, "p": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "-": 0}

whiteking =[
             [ 4,  54,  47, -99, -99,  60,  83, -62 ],
             [-32,  10,  55,  56,  56,  55,  10,   3],
             [-62,  12, -57,  44, -67,  28,  37, -31],
             [-55,  50,  11,  -4, -19,  13,   0, -49],
             [-55, -43, -52, -28, -51, -47,  -8, -50],
             [-47, -42, -43, -79, -64, -32, -29, -32],
             [ -4,   3, -14, -50, -57, -18,  13,   4],
             [ 17,  30,  -3, -14,   6,  -1,  40,  18]
            ]
blackking = [
              [ 17,  30,  -3, -14,   6,  -1,  40,  18],
              [ -4,   3, -14, -50, -57, -18,  13,   4],
              [-47, -42, -43, -79, -64, -32, -29, -32],
              [-55, -43, -52, -28, -51, -47,  -8, -50],
              [-55,  50,  11,  -4, -19,  13,   0, -49],
              [-62,  12, -57,  44, -67,  28,  37, -31],
              [-32,  10,  55,  56,  56,  55,  10,   3],
              [ 4,  54,  47, -99, -99,  60,  83, -62 ]
            ] 
whitequeen= [
              [ 6,   1,  -8,-104,  69,  24,  88,  26],
              [ 14,  32,  60, -10,  20,  76,  57,  24],
              [ -2,  43,  32,  60,  72,  63,  43,   2],
              [  1, -16,  22,  17,  25,  20, -13,  -6],
              [-14, -15,  -2,  -5,  -1, -10, -20, -22],
              [-30,  -6, -13, -11, -16, -11, -16, -27],
              [-36, -18,   0, -19, -15, -15, -21, -38],
              [-39, -30, -31, -13, -31, -36, -34, -42]
            ]

blackqueen = [
               [-39, -30, -31, -13, -31, -36, -34, -42],
               [-36, -18,   0, -19, -15, -15, -21, -38],
               [-30,  -6, -13, -11, -16, -11, -16, -27],
               [-14, -15,  -2,  -5,  -1, -10, -20, -22],
               [  1, -16,  22,  17,  25,  20, -13,  -6],
               [ -2,  43,  32,  60,  72,  63,  43,   2],
               [ 14,  32,  60, -10,  20,  76,  57,  24],
               [ 6,   1,  -8,-104,  69,  24,  88,  26 ]
             ] 

whitebishop = [
                [-59, -78, -82, -76, -23,-107, -37, -50],
                [-11,  20,  35, -42, -39,  31,   2, -22],
                [ -9,  39, -32,  41,  52, -10,  28, -14],
                [ 25,  17,  20,  34,  26,  25,  15,  10],
                [ 13,  10,  17,  23,  17,  16,   0,   7],
                [ 14,  25,  24,  15,   8,  25,  20,  15],
                [ 19,  20,  11,   6,   7,   6,  20,  16],
                [ -7,   2, -15, -12, -14, -15, -10, -10]
              ]

blackbishop = [
                [ -7,   2, -15, -12, -14, -15, -10, -10],
                [ 19,  20,  11,   6,   7,   6,  20,  16],
                [ 14,  25,  24,  15,   8,  25,  20,  15],
                [ 13,  10,  17,  23,  17,  16,   0,   7],
                [ 25,  17,  20,  34,  26,  25,  15,  10],
                [ -9,  39, -32,  41,  52, -10,  28, -14],
                [-11,  20,  35, -42, -39,  31,   2, -22],
                [-59, -78, -82, -76, -23,-107, -37, -50]
              ]

whiteknight = [
                [ -66, -53, -75, -75, -10, -55, -58, -70 ],
                [   -3,  -6, 100, -36,   4,  62,  -4, -14],
                [   10,  67,   1,  74,  73,  27,  62,  -2],
                [   24,  24,  45,  37,  33,  41,  25,  17],
                [   -1,   5,  31,  21,  22,  35,   2,   0],
                [  -18,  10,  13,  22,  18,  15,  11, -14],
                [  -23, -15,   2,   0,   2,   0, -23, -20],
                [  -74, -23, -26, -24, -19, -35, -22, -69]
              ]

blackknight = [
                [  -74, -23, -26, -24, -19, -35, -22, -69],
                [  -23, -15,   2,   0,   2,   0, -23, -20],
                [  -18,  10,  13,  22,  18,  15,  11, -14],
                [   -1,   5,  31,  21,  22,  35,   2,   0],
                [   24,  24,  45,  37,  33,  41,  25,  17],
                [   10,  67,   1,  74,  73,  27,  62,  -2],
                [   -3,  -6, 100, -36,   4,  62,  -4, -14],
                [ -66, -53, -75, -75, -10, -55, -58, -70 ]

              ]

whiterook = [
             [ 35,  29,  33,   4,  37,  33,  56,  50],
             [ 55,  29,  56,  67,  55,  62,  34,  60],
             [ 19,  35,  28,  33,  45,  27,  25,  15],
             [  0,   5,  16,  13,  18,  -4,  -9,  -6],
             [-28, -35, -16, -21, -13, -29, -46, -30],
             [-42, -28, -42, -25, -25, -35, -26, -46],
             [-53, -38, -31, -26, -29, -43, -44, -53],
             [-30, -24, -18,   5,  -2, -18, -31, -32]
            ]  

blackrook = [
              [-30, -24, -18,   5,  -2, -18, -31, -32],
              [-53, -38, -31, -26, -29, -43, -44, -53],
              [-42, -28, -42, -25, -25, -35, -26, -46],
              [-28, -35, -16, -21, -13, -29, -46, -30],
              [  0,   5,  16,  13,  18,  -4,  -9,  -6],
              [ 19,  35,  28,  33,  45,  27,  25,  15],
              [ 55,  29,  56,  67,  55,  62,  34,  60],
              [ 35,  29,  33,   4,  37,  33,  56,  50]
            ]  
whitepawn = [
              [  0,   0,   0,   0,   0,   0,   0,   0],
              [ 78,  83,  86,  73, 102,  82,  85,  90],
              [  7,  29,  21,  44,  40,  31,  44,   7],
              [-17,  16,  -2,  15,  14,   0,  15, -13],
              [-26,   3,  10,   9,   6,   1,   0, -23],
              [-22,   9,   5, -11, -10,  -2,   3, -19],
              [-31,   8,  -7, -37, -36, -14,   3, -31],
              [  0,   0,   0,   0,   0,   0,   0,   0]
            ]  

blackpawn = [
              [  0,   0,   0,   0,   0,   0,   0,   0],
              [-31,   8,  -7, -37, -36, -14,   3, -31],
              [-22,   9,   5, -11, -10,  -2,   3, -19],
              [-26,   3,  10,   9,   6,   1,   0, -23],
              [-17,  16,  -2,  15,  14,   0,  15, -13],
              [  7,  29,  21,  44,  40,  31,  44,   7],
              [ 78,  83,  86,  73, 102,  82,  85,  90],
              [  0,   0,   0,   0,   0,   0,   0,   0]
            ]      

king_endgame = [
                [-10, 0, 10, 10, 10, 10, 0,   -10],
                [-10, 0, 10, 20, 20, 10, 0,   -10],
                [-10, 10, 20, 30, 30, 20, 10, -10],
                [-10, 20, 30, 40, 40, 30, 20, -10],
                [-10, 20, 30, 40, 40, 30, 20, -10],
                [-10, 10, 20, 30, 30, 20, 10, -10],
                [-10, 0, 10, 20, 20, 10, 0,   -10],
                [-10, 0, 10, 10, 10, 10, 0,   -10],
               ]                                              

CHECKMATE = 2000
STALEMATE = 0
DEPTH = 2
QDEPTH = 2
def findRandomMove(validMoves):

    RandomMove = validMoves[random.randint(0, len(validMoves) - 1)]
    return RandomMove

'''
Helper method to make first recursive call.
'''
def findBestMove(gs, validMoves):

    global nextMove
    nextMove = None
    
    def moveOrdering(move):
        gs.makeMove(move)
        Score = scoreMaterial(gs.board)
        gs.undoMove()
        return (-1 if gs.whiteToMove else 1) * Score
    
    validMoves.sort(key=moveOrdering)

    findMoveNegaMaxAlphaBeta(gs, validMoves, DEPTH, -CHECKMATE, CHECKMATE, 1 if gs.whiteToMove else -1)
    return nextMove 

def findMoveMinMax(gs, validMoves, depth, whiteToMove):
    global nextMove

    if depth == 0 :
        return scoreBoard(gs)

    if whiteToMove :
        maxScore = -CHECKMATE
        random.shuffle(validMoves)
        for move in validMoves :
            gs.makeMove(move)
            nextMoves = gs.getValidMoves()
            score = findMoveMinMax(gs, nextMoves, depth - 1, False)
            if score > maxScore :
                maxScore = score
                if depth == DEPTH:
                    nextMove = move
            gs.undoMove()  
        return maxScore          

    else :
        minScore = CHECKMATE
        random.shuffle(validMoves)
        for move in validMoves :
            gs.makeMove(move)
            nextMoves = gs.getValidMoves()
            score = findMoveMinMax(gs, nextMoves, depth - 1, True)
            if score < minScore:
                minScore = score
                if depth == DEPTH:
                    nextMove = move
            gs.undoMove()
        return minScore      

def findMoveNegaMaxAlphaBeta(gs, validMoves, depth, alpha, beta, turnMultiplier):
    global nextMove

    if depth == 0 :
        requiredMoves = gs.getValidMoves()
        actual_score = quiescienceSearch(gs, requiredMoves, -CHECKMATE, CHECKMATE, 1 if gs.whiteToMove else -1, depth= QDEPTH)
        return actual_score
    maxScore = -CHECKMATE
    for move in validMoves :
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()
        score = -findMoveNegaMaxAlphaBeta(gs, nextMoves, depth - 1 , -beta, -alpha, -turnMultiplier)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH :
                nextMove =  move
        gs.undoMove() 
        if maxScore > alpha:   # This is were pruning happens
            alpha = maxScore
        if alpha >= beta :
            break    
    return maxScore            

def quiescienceSearch(gs, validMoves, alpha, beta, turnmultiplier, depth):
    quiescienceMoves = []
    for move in validMoves:
        if gs.inCheck():
            quiescienceMoves.append(move)
        if "x" in move.getChessNotation():
            quiescienceMoves.append(move)
        if move.isPawnPromotion :
            quiescienceMoves.append(move)
    if len(quiescienceMoves) == 0 or depth == 0:
        return turnmultiplier * scoreBoard(gs)   
    
    def quiescenceMoveOrderdering(move):
        ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,
                   "5": 3, "6": 2, "7": 1, "8": 0}
        filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, 
                   "e": 4, "f": 5, "g": 6, "h": 7}
        values = { "p" : 1, "R" : 5, "B" : 3, "N" : 3, "Q" : 9, "-" : 0 , "K" : 0}
        pieceMoved = move.pieceMoved[1]
        pieceMovedValue = values[pieceMoved]
        pieceCaptured = move.pieceCaptured[1]
        pieceCapturedValue = values[pieceCaptured]
        return pieceCapturedValue - pieceMovedValue

    quiescienceMoves.sort(key=quiescenceMoveOrderdering)


    maxScore = turnmultiplier * scoreBoard(gs)
    for move in quiescienceMoves :
        gs.makeMove(move)
        nextMoves = gs.getValidMoves()
        score = -quiescienceSearch(gs, nextMoves, -beta, -alpha, -turnmultiplier, depth - 1)
        if score > maxScore:
            maxScore = score 
        gs.undoMove()
        if maxScore > alpha :
            alpha = maxScore
        if alpha >= beta :
            break
    
    return maxScore

'''
Score the board based on material
'''
def scoreMaterial(board):
    score = 0
    for row in board:
        for square in row :
            if square[0] == "w":
                score += pieceScore[square[1]]
            elif square[0] == "b":
                score -= pieceScore[square[1]]
    return score    

def scoreBoard(gs):

    draw = False
    
    for log in gs.boardLog :
        if gs.boardLog.count(log) > 2 :
            draw = True

    if gs.checkMate:
        if gs.whiteToMove :
            return -CHECKMATE    # Black wins
        else :
            return CHECKMATE    # White wins
    elif gs.staleMate:
        return STALEMATE    # Neither side wins 

    elif draw :
        return STALEMATE    

    score = 0
    for row in range(8):
        for col in range(8) :
            piece = gs.board[row][col]
            if piece[0] == "w" :
                if piece[1] == "p":
                    score += (pieceScore[piece[1]] + whitepawn[row][col])
                elif piece[1] == "K":
                    score += whiteking[row][col]
                elif piece[1] == "Q":
                    score += (pieceScore[piece[1]] + whitequeen[row][col])
                elif piece[1] == "R":
                    score += (pieceScore[piece[1]] + whiterook[row][col])
                elif piece[1] == "B":
                    score += (pieceScore[piece[1]] + whitebishop[row][col])
                elif piece[1] == "N":
                    score += (pieceScore[piece[1]] + whiteknight[row][col]) 
                else :
                    continue                     
            else :
                if piece[1] == "p":
                    score += (pieceScore[piece[1]] + blackpawn[row][col])
                elif piece[1] == "K":
                    score += blackking[row][col]
                elif piece[1] == "Q":
                    score += (pieceScore[piece[1]] + blackqueen[row][col])
                elif piece[1] == "R":
                    score += (pieceScore[piece[1]] + blackrook[row][col])
                elif piece[1] == "B":
                    score += (pieceScore[piece[1]] + blackbishop[row][col])
                elif piece[1] == "N":
                    score += (pieceScore[piece[1]] + blackknight[row][col])
                else :
                    continue

    finalScore = 0
    for row in range(8):
        for col in range(8) :
            piece = gs.board[row][col]
            if piece[0] == "w" :
                if piece[1] == "p":
                    finalScore += (pieceScore[piece[1]] + whitepawn[row][col])
                elif piece[1] == "K":
                    if score >= 4600 :
                        finalScore += whiteking[row][col]
                    else :
                        finalScore += king_endgame[row][col]    
                elif piece[1] == "Q":
                    finalScore += (pieceScore[piece[1]] + whitequeen[row][col])
                elif piece[1] == "R":
                    finalScore += (pieceScore[piece[1]] + whiterook[row][col])
                elif piece[1] == "B":
                    finalScore += (pieceScore[piece[1]] + whitebishop[row][col])
                elif piece[1] == "N":
                    finalScore += (pieceScore[piece[1]] + whiteknight[row][col]) 
                else :
                    continue                     
            else :
                if piece[1] == "p":
                    finalScore -= (pieceScore[piece[1]] + blackpawn[row][col])
                elif piece[1] == "K":
                    if score >= 4600 :
                        finalScore -= blackking[row][col]
                    else :
                        finalScore -= king_endgame[row][col]    
                elif piece[1] == "Q":
                    finalScore -= (pieceScore[piece[1]] + blackqueen[row][col])
                elif piece[1] == "R":
                    finalScore -= (pieceScore[piece[1]] + blackrook[row][col])
                elif piece[1] == "B":
                    finalScore -= (pieceScore[piece[1]] + blackbishop[row][col])
                elif piece[1] == "N":
                    finalScore -= (pieceScore[piece[1]] + blackknight[row][col])
                else :
                    continue                
                    
    return finalScore    