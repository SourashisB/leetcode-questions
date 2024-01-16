class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        def validate(row,column,number):
            for i in range(0,9):
                if board[i][column] == number:
                    return False
                if board[row][i] == number:
                    return False
                if board[3*(row//3) + i//3][3*(column//3) + i%3] == number:
                    return False
            
            return True
        
        def fillBoard(row, column):
            if row == 9:
                return True
            if column == 9:
                return (fillBoard(row+1,0))
            if board[row][column] ==".":
                for i in range(1,10):
                    if validate(row,column,str(i)) ==True:
                        board[row][column]=str(i)
                        if fillBoard(row,column+1):
                            return True
                        board[row][column]="."
                return False
            return fillBoard(row,column+1)
        
            
        fillBoard(0,0)
                                   
            






test = Solution()

board = [["5","3",".",".","7",".",".",".","."],
         ["6",".",".","1","9","5",".",".","."],
         [".","9","8",".",".",".",".","6","."],
         ["8",".",".",".","6",".",".",".","3"],
         ["4",".",".","8",".","3",".",".","1"],
         ["7",".",".",".","2",".",".",".","6"],
         [".","6",".",".",".",".","2","8","."],
         [".",".",".","4","1","9",".",".","5"],
         [".",".",".",".","8",".",".","7","9"]]

print(test.solveSudoku(board))
print(board)