function [theta,j,it_n]=Gradient_Descent_Algorithm(x,y,theta,paras)
%Gradient descent method for univariety
%Harbin Institute of Technology
%Swiftie233
%date:2021/5/26

%% debug
% x=[1,2,3,4,5];
% y=[1,2,3,4,5];
% m=length(x);
% alpha=0.15;
% theta0=[0];
% theta1=[0];

% eps=10^-6;
% it_limits=1000;
%% Initialization
alpha=paras(1);
eps=paras(2);
it_limits=paras(3);
m=length(x);
theta0=theta(1);
theta1=theta(2);

j=[];
it_n=0;
%% GD algorithm


while(1)
    hx=theta0(end)+theta1(end).*x;
    j=1/(2*m)*sum((hx-y).^2);
    dj0=1/m*sum(hx-y);
    dj1=1/m*sum((hx-y).*x);

    temp0=theta0(end)-alpha*dj0;
    temp1=theta1(end)-alpha*dj1;
    theta0=[theta0,temp0];
    theta1=[theta1,temp1];
    it_n=it_n+1;
    
    flag0=abs(theta0(end)-theta0(end-1));
    flag1=abs(theta1(end)-theta1(end-1));
    flag=flag0<=eps&&flag1<=eps;
    
    if flag||it_n>=it_limits
        break
    end
    
end
theta=[theta0(end),theta1(end)]';
