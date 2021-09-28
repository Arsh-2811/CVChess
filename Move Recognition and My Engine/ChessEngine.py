import io

# This is class is responsible for storing all the information about the current state of a chess game. It will also be resposible for determining the valid moves at the current state.
# It will also keep a move log.

class GameState():

    def __init__(self):
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        ]

        self.moveFunctions = {"p": self.getPawnMoves,"R": self.getRookMoves,"N": self.getKnightMoves,
                              "B": self.getBishopMoves, "Q": self.getQueenMoves, "K": self.getKingMoves}

        self.whiteToMove = True
        self.moveLog = []
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)
        self.checkMate = False
        self.staleMate = False
        self.enpassantPossible = ()   # Coordinate for the square where enpassant is possible
        self.enpassantPossibleLog = [self.enpassantPossible]
        self.currentCastlingRights = CastleRights(True, True, True, True)
        self.castleRightsLog = [CastleRights(self.currentCastlingRights.wks, self.currentCastlingRights.bks, 
                                             self.currentCastlingRights.wqs, self.currentCastlingRights.bqs)]
        self.boardLog = []                                     

    '''
    Takes a move as a parameter and executes it (this will not work for castling , en-passant and pawn promotion) .
    '''
    def makeMove(self, move):
        self.board[move.startRow][move.startCol]  = "--"
        self.board[move.endRow][move.endCol] = move.pieceMoved
        self.moveLog.append(move) # log the move in order to display the history of the game (later)
        self.whiteToMove = not self.whiteToMove # Swap the players

        '''
        Update the king's location if a move is made
        '''
        if move.pieceMoved == "wK" :
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved == "bK" :
            self.blackKingLocation = (move.endRow, move.endCol) 

        '''
        Pawn Promotion
        '''    
        if move.isPawnPromotion :
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + "Q"

        '''
        EnPassant Capture
        '''    
        if move.isEnpassantMove :
            self.board[move.startRow][move.endCol] = "--"    # Capturing the pawn

        '''
        Updating enpassantPossible variable
        '''    
        if move.pieceMoved[1] == "p" and abs(move.startRow - move.endRow) == 2 :   # Only on 2 square pawn advances
            self.enpassantPossible = ((move.startRow + move.endRow)//2, move.endCol)
        else :
            self.enpassantPossible = ()   

        self.enpassantPossibleLog.append(self.enpassantPossible)    
 
        '''
        Castle Move
        '''
        self.updateCastleRights(move)
        self.castleRightsLog.append(CastleRights(self.currentCastlingRights.wks, self.currentCastlingRights.bks, 
                                             self.currentCastlingRights.wqs, self.currentCastlingRights.bqs))                                   

        if move.isCastleMove :
            if move.endCol - move.startCol == 2 :    # King side castle
                self.board[move.endRow][move.endCol - 1] = self.board[move.endRow][move.endCol + 1]    # Moves the rook
                self.board[move.endRow][move.endCol + 1] = "--"    # Erase the old Rook

            else:    # Queen Side Castle Move
                self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][move.endCol - 2]    # Moves the rook
                self.board[move.endRow][move.endCol - 2] = "--"    # Erase the old Rook

        '''
        Threefold Repetition by checking the FEN Value
        '''  
        def board_to_fen(board):
            with io.StringIO() as s:
                for row in board:
                    empty = 0
                    for cell in row:
                        c = cell[0]
                        if c in ('w', 'b'):
                            if empty > 0:
                                s.write(str(empty))
                                empty = 0
                            s.write(cell[1].upper() if c == 'w' else cell[1].lower())
                        else:
                            empty += 1
                    if empty > 0:
                        s.write(str(empty))
                    s.write('/')
                s.seek(s.tell() - 1)
                s.write(' w KQkq - 0 1')
                return s.getvalue()    

        self.boardLog.append(board_to_fen(self.board))    
                                   
    '''
    Undo the last move made
    '''    
    def undoMove(self):
        if len(self.moveLog) != 0 :  # Make sure that there is a move to undo
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove # Switch turns back
            
            '''
            Update king's location if we undo a move
            '''
            if move.pieceMoved == "wK" :
                self.whiteKingLocation = (move.startRow, move.startCol)
            elif move.pieceMoved == "bK" :
                self.blackKingLocation = (move.startRow, move.startCol) 

            '''
            Undo enpassant move
            '''
            if move.isEnpassantMove:
                self.board[move.endRow][move.endCol] = "--"   # Leave landing square blank
                self.board[move.startRow][move.endCol] = move.pieceCaptured
                
            self.enpassantPossibleLog.pop()
            self.enpassantPossible = self.enpassantPossibleLog[-1]    

            '''
            Undo Castling Rights
            '''    
            self.castleRightsLog.pop()    # Get rid of the new castle rights from the move we are undoing
            self.currentCastlingRights.wks = self.castleRightsLog[-1].wks    # Set the current castle rights to the last one in the list
            self.currentCastlingRights.bks = self.castleRightsLog[-1].bks
            self.currentCastlingRights.wqs = self.castleRightsLog[-1].wqs
            self.currentCastlingRights.bqs = self.castleRightsLog[-1].bqs

            '''
            Undo the Castle Move
            '''
            if move.isCastleMove:
                if move.endCol - move.startCol == 2 :
                    self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][move.endCol - 1]
                    self.board[move.endRow][move.endCol - 1] = "--"
                else :
                    self.board[move.endRow][move.endCol - 2] = self.board[move.endRow][move.endCol + 1]
                    self.board[move.endRow][move.endCol + 1] = "--"

            self.checkMate = False      

            self.staleMate = False 

            self.boardLog.pop() 
    '''
    Update the castle rights given the move
    '''
    def updateCastleRights(self, move):
        if move.pieceMoved == "wK":
            self.currentCastlingRights.wks = False
            self.currentCastlingRights.wqs = False
        elif move.pieceMoved == "bK":
            self.currentCastlingRights.bks = False
            self.currentCastlingRights.bqs = False
        elif move.pieceMoved == "wR":
            if move.startCol == 7:
                if move.startCol == 0:
                    self.currentCastlingRights.wqs == False
                elif move.endCol == 7:
                    self.currentCastlingRights.wks = False
        elif move.pieceMoved == "bR":
            if move.startCol == 0:
                if move.startCol == 0:
                    self.currentCastlingRights.wqs == False
                elif move.endCol == 7:
                    self.currentCastlingRights.wks = False  
        if move.pieceCaptured == 'wR':
            if move.endRow == 7:
                if move.endCol == 0:
                    self.currentCastlingRights.wqs = False
                elif move.endCol == 7:
                    self.currentCastlingRights.wks = False
        elif move.pieceCaptured == 'bR':
            if move.endRow == 0:
                if move.endCol == 0:
                    self.currentCastlingRights.bqs = False
                elif move.endCol == 7:
                    self.currentCastlingRights.bks = False                                 

    '''
    All moves considering checks.
    '''        
    def getValidMoves(self):
        
        tempEnpassantPossible = self.enpassantPossible
        tempCastlingRights = CastleRights(self.currentCastlingRights.wks, self.currentCastlingRights.bks, 
                                          self.currentCastlingRights.wqs, self.currentCastlingRights.bqs)
        tempBoardLog = self.boardLog  

        #1.) Generate all possible moves
        #2.) For each move , make the move 
        #3.) Generate all opponent's move
        #4.) For each of your opponent's move , if they attack your king
        #5.) If they attack your king , it is not a valid move

        moves = self.getAllPossibleMoves()

        if self.whiteToMove:
            self.getCastleMoves(self.whiteKingLocation[0], self.whiteKingLocation[1], moves)
        else :
            self.getCastleMoves(self.blackKingLocation[0], self.blackKingLocation[1], moves)    

        for i in range(len(moves)-1, -1, -1):      # When removing from a list go backwards through that list
            self.makeMove(moves[i])

            self.whiteToMove = not self.whiteToMove   # We have to switch the move because our makeMove function switched the turns earlier
            if self.inCheck():
                moves.remove(moves[i])        
            self.whiteToMove = not self.whiteToMove
            self.undoMove()   

        if len(moves) == 0 :   # This means either checkmate or stalemate
            if self.inCheck():
                self.checkMate = True
            else :
                self.staleMate = True
        else :
            self.checkMate = False
            self.staleMate = False 

        self.enpassantPossible = tempEnpassantPossible 
        self.currentCastlingRights = tempCastlingRights 

        self.boardLog = tempBoardLog         
        
        return moves

    '''
    Determine if the current player is in check
    '''
    def inCheck(self):
        if self.whiteToMove:
            return self.squareUnderAttack(self.whiteKingLocation[0], self.whiteKingLocation[1])
        else:
            return self.squareUnderAttack(self.blackKingLocation[0], self.blackKingLocation[1])    
    '''
    Determine if the enemy can attack the square (r, c)
    '''
    def squareUnderAttack(self, r, c) :

        self.whiteToMove = not self.whiteToMove    # Switch to opponent's move to see their point of view
        oppMoves = self.getAllPossibleMoves()
        self.whiteToMove = not self.whiteToMove
        for move in oppMoves:
            if move.endRow == r and move.endCol == c :   # Square is under attack
                return True
        return False


    '''
    All moves without considering checks.
    '''
    def getAllPossibleMoves(self):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]         # First character of the piece on the given square. It will be either w , b or - (which means there is no piece on that square).
                if (turn == "w" and self.whiteToMove) or (turn == "b" and not self.whiteToMove):
                    piece = self.board[r][c][1]

                    self.moveFunctions[piece](r, c, moves)   # Calls the appropriate move function based on piece type

        return moves                

    '''
    Get all the pawn moves for the pawn located at row, col and these moves to the list
    '''
    def getPawnMoves(self, r, c, moves):

        if self.whiteToMove :       # Whie pawn moves
            if self.board[r-1][c] == "--" :    # 1 square pawn advance
                moves.append(Move((r, c), (r-1, c), self.board))
                if r==6 and self.board[r-2][c] == "--" : # 2 square pawn advance
                    moves.append(Move((r, c), (r-2, c), self.board))

            if c-1 >= 0 :    # Captures to the left
                if self.board[r-1][c-1][0] == "b" :   # Make sure that there is a enemy piece to capture
                    moves.append(Move((r, c), (r-1, c-1), self.board))
                elif (r-1, c-1) == self.enpassantPossible :
                    moves.append(Move((r, c), (r-1, c-1), self.board, isEnpassantMove = True))

            if c+1 <= 7:   # Captures to the right
                if self.board[r-1][c+1][0] == "b" :
                    moves.append(Move((r, c), (r-1, c+1), self.board))
                elif (r-1, c+1) == self.enpassantPossible :
                    moves.append(Move((r, c), (r-1, c+1), self.board, isEnpassantMove = True))    

        else :    # Black pawn moves

            if self.board[r+1][c] == "--" :    # 1 square pawn advance
                moves.append(Move((r, c), (r+1, c), self.board))
                if r==1 and self.board[r+2][c] == "--" : # 2 square pawn advance
                    moves.append(Move((r, c), (r+2, c), self.board))

            if c-1 >= 0 :    # Captures to the left
                if self.board[r+1][c-1][0] == "w" :   # Make sure that there is a enemy piece to capture
                    moves.append(Move((r, c), (r+1, c-1), self.board))
                elif (r+1, c-1) == self.enpassantPossible :
                    moves.append(Move((r, c), (r+1, c-1), self.board, isEnpassantMove = True))    

            if c+1 <= 7:   # Captures to the right
                if self.board[r+1][c+1][0] == "w" :
                    moves.append(Move((r, c), (r+1, c+1), self.board)) 
                elif (r+1, c+1) == self.enpassantPossible :
                    moves.append(Move((r, c), (r+1, c+1), self.board, isEnpassantMove = True))      

        # Add pawn promotions move later                                 
                        
    
    '''
    Get all the rook moves for the rook located at row, col and these moves to the list
    '''
    def getRookMoves(self, r, c, moves):
        directions = ((-1, 0), (0, -1), (1, 0), (0, 1))
        enemyColor = "b" if self.whiteToMove else "w"
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8 :    # On Board
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--":  # empty space valid
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor :   # enemy piece valid
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else :    # friendly piece invalid
                        break
                else :      # off board
                    break    
    
    '''
    Get all the knight moves for the knight located at row, col and these moves to the list
    '''
    def getKnightMoves(self, r, c, moves):
        knightMoves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, 1), (2, -1))
        allyColor = "w" if self.whiteToMove else "b"
        for m in knightMoves :
            endRow = r + m[0]
            endCol = c + m[1]
            if 0<= endRow < 8 and 0 <= endCol < 8 :
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor : #not an ally piece
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    '''
    Get all the bishop moves for the bishop located at row, col and these moves to the list
    '''
    def getBishopMoves(self, r, c, moves):
        directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        enemyColor = "b" if self.whiteToMove else "w"
        for d in directions:
            for i in range(1, 8):
                endRow = r + d[0] * i
                endCol = c + d[1] * i
                if 0 <= endRow < 8 and 0 <= endCol < 8 :    # On Board
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--":  # empty space valid
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor :   # enemy piece valid
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else :    # friendly piece invalid
                        break
                else :      # off board
                    break 
    
    '''
    Get all the queen moves for the queen located at row, col and these moves to the list
    '''
    def getQueenMoves(self, r, c, moves):
        self.getRookMoves(r, c, moves)
        self.getBishopMoves(r, c,moves)
    
    '''
    Get all the king moves for the king located at row, col and these moves to the list
    '''
    def getKingMoves(self, r, c, moves):
        kingMoves  = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        allyColor = "w" if self.whiteToMove else "b"
        for i in range(8):
            endRow = r + kingMoves[i][0]
            endCol = c + kingMoves[i][1]
            if 0 <= endRow < 8 and 0 <= endCol < 8 :
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor :     # Not an ally piece
                    moves.append(Move((r, c), (endRow, endCol), self.board))

    '''
    Generate all valid catle moves for the king at (r, c) and add them to the list of moves
    '''                
    def getCastleMoves(self, r, c, moves):
        if self.squareUnderAttack(r, c):
            return    # Can't castle when the king is in check
        if (self.whiteToMove and self.currentCastlingRights.wks) or (not self.whiteToMove and self.currentCastlingRights.bks):
            self.getKingSideCastleMoves(r, c, moves)   
        if (self.whiteToMove and self.currentCastlingRights.wqs) or (not self.whiteToMove and self.currentCastlingRights.bqs):
            self.getQueenSideCastleMoves(r, c, moves)    

    def getKingSideCastleMoves(self, r, c, moves):
        if self.board[r][c+1] == "--" and self.board[r][c+2] == "--":
            if not self.squareUnderAttack(r, c+1) and not self.squareUnderAttack(r, c+2):
                moves.append(Move((r, c), (r, c+2), self.board, isCastleMove=True))

    def getQueenSideCastleMoves(self, r, c, moves):
        if self.board[r][c-1] == "--" and self.board[r][c-2] == "--" and self.board[r][c-3] == "--" :
            if not self.squareUnderAttack(r, c-1) and not self.squareUnderAttack(r, c-2) :
                moves.append(Move((r, c), (r, c-2), self.board, isCastleMove=True))
    

class CastleRights():
    def __init__(self, wks, bks, wqs, bqs):
        self.wks = wks
        self.bks = bks
        self.wqs = wqs
        self.bqs = bqs

class Move():

    # maps keys to values
    # key : value
    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,
                   "5": 3, "6": 2, "7": 1, "8": 0}

    rowsToRanks = {v: k for k, v in ranksToRows.items()}

    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3, 
                   "e": 4, "f": 5, "g": 6, "h": 7}

    colsToFiles = {v: k for k, v in filesToCols.items()}                              

    def __init__(self, startSq, endSq, board, isEnpassantMove = False, isCastleMove = False):
        self.startRow = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]

        '''
        This is for Pawn Promotion
        '''
        self.isPawnPromotion = False

        if (self.pieceMoved == "wp" and self.endRow == 0) or (self.pieceMoved == "bp" and self.endRow == 7) :
            self.isPawnPromotion = True

        '''
        This is for EnPassant Capture
        '''
        self.isEnpassantMove = isEnpassantMove

        if self.isEnpassantMove :
            self.pieceCaptured = "wp" if self.pieceCaptured == "wp" else "bp"

        self.isCastleMove = isCastleMove    

        '''
        This will give us a Unique ID for every move which can be then used to equate 2 moves like if we have to check that the move we made is in the list of valid moves or not.
        '''
        self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol   
    '''
    Overriding the equals method
    '''
    def __eq__(self, other):
        if isinstance(other, Move):
            return self.moveID == other.moveID
        return False

    def getChessNotation(self):
        output_string = ""
        if self.isPawnPromotion:
            output_string += self.getRankFile(self.endRow, self.endCol) + "=Q"
        elif self.isCastleMove:
            if self.endCol == 2 :
                output_string += "O-O-O"    
            else :
                output_string += "O-O"  
        elif self.isEnpassantMove:
            output_string += self.getRankFile(self.startRow, self.startCol)[0] + "x" + self.getRankFile(self.endRow, self.endCol) 
        elif self.pieceCaptured != "--":
            if self.pieceMoved[1] == "p":
                output_string += self.getRankFile(self.startRow, self.startCol)[0] + "x" + self.getRankFile(self.endRow, self.endCol)
            else :
                output_string += self.pieceMoved[1] + "x" + self.getRankFile(self.endRow, self.endCol)      
        else:
            if self.pieceMoved[1] == "p":
                output_string += self.getRankFile(self.endRow, self.endCol)
            else :
                output_string += self.pieceMoved[1] + self.getRankFile(self.endRow, self.endCol)
        return output_string                              

    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]    