function [theta,j,it_n,delta]=Gradient_Descent_Algorithm_MatrixVersion(x,y,theta,paras)
%Normal Gradient Descent Algorithm for multivarieties
%Harbin Institute of Technology
%Swiftie233
%date:2021/5/26

%% Debug
% clear 
% clc
%f(x)=2+x1+2x2+3x3
%theta=[2 1 2 3]'
% x=[1 1 1 1
%  1 2  2 2
%  1 3  3 3];
% y=[8 14 20]';
% 
% alpha=0.1;
% eps=10^-3;
% it_limits=1000;
% it_n=0;
% theta=zeros(n,1);
%% Initialization
alpha=paras(1);
eps=paras(2);
it_limits=paras(3);

n=length(x(1,:));
m=length(x(:,1));

j=[];
dj=zeros(n,1);
it_n=0;
%% GD Algorithm

while(1)
    h=x*theta(:,it_n+1);
    delta=h-y;
    j=[j,1/(2*m)*sum(delta.^2)];             %cost函数
    
    for i = 1:n
        dj(i)=1/m*sum(delta.*x(:,i));        %对theta偏导向量
    end
    theta=[theta theta(:,it_n+1)-alpha*dj];
    it_n=it_n+1;
    
    temp=abs(theta(:,end)-theta(:,end-1));   %两次迭代theta的差值
    flag=sum(temp<eps)==n;                   %统计n个theta值的最后两次迭代差值是否全部小于eps
    if (it_n>=it_limits)||flag               %迭代次数达到限制或满足迭代差时退出
        break
    end
    
end