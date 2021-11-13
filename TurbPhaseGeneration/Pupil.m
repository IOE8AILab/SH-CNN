% 子函数 Pupil() 用于
function Pxy= Pupil(Na,d)
% 产生归一化的瞳函数，通过改变输入d的值可以将瞳函数归一化到不同口径的圆域上
error(nargchk(1,2,nargin));
if nargin<2, d=1; end
[xx0,yy0] = meshgrid(linspace(-1,1,Na));
Pxy = zeros(Na);
Pxy(xx0.^2+yy0.^2<=1)=d;
