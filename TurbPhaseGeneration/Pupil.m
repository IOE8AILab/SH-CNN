% �Ӻ��� Pupil() ����
function Pxy= Pupil(Na,d)
% ������һ����ͫ������ͨ���ı�����d��ֵ���Խ�ͫ������һ������ͬ�ھ���Բ����
error(nargchk(1,2,nargin));
if nargin<2, d=1; end
[xx0,yy0] = meshgrid(linspace(-1,1,Na));
Pxy = zeros(Na);
Pxy(xx0.^2+yy0.^2<=1)=d;
