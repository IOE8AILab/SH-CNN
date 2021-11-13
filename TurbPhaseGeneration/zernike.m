%%%%%%%求第N阶的zernik多项式z，半径采用单位圆规一化，Na为采样点数，根据zernik多项式定义式编写
%% this function is used to draw the phase map of a Zernike mode
%% created by Jian Huang
%% mode: the Zernike mode; Na: the grid size of the phase map
%% z: the phase map
function z=zernike(mode,Na)
    [n,m]=nmzern(mode);
    x=linspace(-1,1,Na);
    y=x;
    [x,y]=meshgrid(x,y);
    r=sqrt(x.^2+y.^2);
    th=atan2(y,x);
    s=0;
    R=0;
    while (s>=0)&&(s<=(n-m)/2)
        a=(-1)^s*factorial((n-s));
        b=factorial(s)*factorial(((n+m)/2-s))*factorial(((n-m)/2-s));
        k=a/b;
        R=R+k.*r.^(n-2*s);
        s=s+1;
    end
    if m==0
        z=sqrt(n+1)*R;
    else
        if mod(mode,2)==0
            z=sqrt(2*(n+1))*R.*cos(m*th);
        else
            z=sqrt(2*(n+1))*R.*sin(m*th);
        end
    end
    z=z.*Pupil(Na);%Pupil为圆域截断函数
return