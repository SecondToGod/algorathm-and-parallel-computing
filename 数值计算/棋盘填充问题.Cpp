#include<iostream>
using namespace std;

int tile=1;                   //骨牌编号
int board[200][200];  //棋盘
/*
* x-棋盘左上角行号
* y-棋盘左上角列号
* m-特殊方格行号
* n-特殊方格列号
*/
void chessBoard ( int x, int y, int m, int n, int size)
{
    if ( size==1 )    //棋盘方格大小为1时递归到最里层
        return;
    int t=tile++;     //每次递增1
    int s =size/2;    //划分
    //检查左上角子棋盘中
    if ( m<x+s && n<y+s )              //在
        chessBoard ( x, y, m, n, s );
    else         
    {
        board[x+s-1][y+s-1]=t;       //不在,填充该子棋盘右下角的方块
        chessBoard ( x, y, x+s-1, y+s-1, s );
    }
    //检查右上角子棋盘中
    if ( m<x+s && n>=y+s )               //在
        chessBoard ( x, y+s, m, n, s );
    else          
    {
        board[x+s-1][y+s]=t;        //不在，填充该子棋盘左下角的方块
        chessBoard ( x, y+s, x+s-1, y+s, s );
    }
    //检查左下角子棋盘中
    if ( m>=x+s && n<y+s )              //在
        chessBoard ( x+s, y, m, n, s );
    else            
    {
        board[x+s][y+s-1]=t;         //不在，填充该子棋盘右上角的方块
        chessBoard ( x+s, y, x+s, y+s-1, s );
    }
    //检查右下角子棋盘中
    if ( m>=x+s && n>=y+s )                //在
        chessBoard ( x+s, y+s, m, n, s );
    else         
    {
        board[x+s][y+s]=t;           //不在，填充该子棋盘左上角的方块
        chessBoard ( x+s, y+s, x+s, y+s, s );
    }
}
int main()
{
    int size;
    cout<<"输入棋盘的size（2^k）: \n";
    cin>>size;
    int p,q;          
    cout<<"输入特殊方格位置的坐标: \n"; //定义特殊点位置
    cin>>p>>q;
    chessBoard (0,0,p-1,q-1,size);
	 for (int i=0; i<size; i++)
    {
        for (int j=0; j<size; j++)
            cout<<board[i][j]<<"   ";
        cout<<endl;
    }
    return 0;
}