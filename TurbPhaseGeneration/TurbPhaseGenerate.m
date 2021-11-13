%% This code is used to generate the training data used by "Deep Phase Retrieval for Astronomical Shack-Hartmann Wavefront Sensor"
%% submitted to MNRAS
%% 2021-08-12, Youming Guo, Institue of Optics and Electronics, Chinese Academy of Sciences.


clear;clc;close all;
%% Parameter initialization
HSLoc = load('Parafiles\HSWinLT_100.txt');% load the configuration of the SHWFS
Pixel_Num = 240;%Camera Format
Pixel_Size = 24e-6;%pixel size
FrameNum = 100000;% number of phases used to train the CNN
NmodeTotal = 299;% Zernike modes to be reconstructed by the CNN
Xloc = HSLoc(:,1);
Yloc = Pixel_Num - HSLoc(:,2);
Xloc = Xloc.*Pixel_Size;
Yloc = Yloc.*Pixel_Size;
Nsub = length(Xloc);% number of subapertures
Sub_Num = 12;%the SHWFS is a size of 12x12
Pixel_Lens = 20;%number of pixel in each subaperture
f = 16.6e-3;%focal length of the microlens
dx = 20.*Pixel_Size;% the image size of the subaperture
dy = dx;
r = 6*dx;
wl = 658e-9;% wavelength
kc = 2.*pi./wl;
telediam = 1.8;% diameter of the telescope
r0 = 0.1; % Fried parameter


%% this section is used to build the SHWFS model
Na = Sub_Num*Pixel_Lens;
Pxy_Circle = Pupil(Na,Na)./Na;
dPixel_Lens = Pixel_Size.*Pixel_Num/Sub_Num/Pixel_Lens;
x1 = dPixel_Lens:dPixel_Lens:Pixel_Size*Pixel_Num;
y1 = dPixel_Lens:dPixel_Lens:Pixel_Size*Pixel_Num;

[X2,Y2] = meshgrid(x1,y1);
PhaseMask = zeros(Na,Na);
PhaseMask_3D = zeros(Na,Na,Nsub);
Pxy = zeros(Na,Na);
for i = 1:Nsub
    PhaseMask = zeros(Na,Na);
    x0 = Xloc(i) + dx/2;
    y0 = Yloc(i) - dy/2;
    % pupil of the subapertures
    PhaseMask(round((Yloc(i)-dx+dPixel_Lens)/dPixel_Lens):round((Yloc(i))/dPixel_Lens),...
        round((Xloc(i)+dPixel_Lens)/dPixel_Lens):round((Xloc(i)+dx)/dPixel_Lens)) = 1;
    %pupil of the SHWFS
    Pxy(round((Yloc(i)-dx+dPixel_Lens)/dPixel_Lens):round((Yloc(i))/dPixel_Lens),...
        round((Xloc(i)+dPixel_Lens)/dPixel_Lens):round((Xloc(i)+dx)/dPixel_Lens)) = 1;
    PhaseMask = flip(PhaseMask,1);
    PhaseMask_3D(:,:,i) = PhaseMask;

end

%% this section is used to generate the turbulence distorted wavefront
Phase_Aberration_DataSet = zeros(Na*Na,NmodeTotal);% the phase map of Zernike modes
for nmode = 1:NmodeTotal
    Phase_Aberration_DataSet(:,nmode) = reshape(zernike(nmode+1,Na),Na*Na,1);
end

Pxy = Pxy.*Pxy_Circle;


Zer_DataSet = ZernikeSerial(telediam,r0,NmodeTotal,FrameNum);
Zer_DataSet = Zer_DataSet';

%% this section is used to simulate the SHWFS sensing procedure
%% and get the correspongding SHWFS images of the distorted wavefront

nrepeat = 1000; % Save as a .mat file every 1000 images
ISHWFS_3D = zeros(Na*Na,nrepeat);
Zer = zeros(NmodeTotal,nrepeat);
nk = 0;
k = 0;
for nnum = 1:FrameNum
    k = k + 1;
    disp(k)
    
    Phase_Aberration = reshape(Phase_Aberration_DataSet*Zer_DataSet(:,nnum),Na,Na);
    Uin = Pxy.*exp(1i.*Phase_Aberration);
    IFar = zeros(Na,Na);
    for i = 1:Nsub
%         i
        USubTemp = Uin(PhaseMask_3D(:,:,i)==1);
        USubTemp = reshape(USubTemp,Pixel_Lens,Pixel_Lens);
        ISubFarTemp = abs(fftshift(fft2(USubTemp))).^2; % compute the far field image of the subapertures using fft2
        IFar(PhaseMask_3D(:,:,i)==1) = ISubFarTemp;
    end
    
    figure(101);imagesc(reshape(IFar,Na,Na));colorbar;pause(0.1);
    ISHWFS_3D(:,k) = reshape(IFar,Na*Na,1);
    Zer(:,k) = Zer_DataSet(:,nnum);
    if k == nrepeat
        nk = nk + 1;
        %***********save the data with mat file************
        str = strcat('save DataSet\ISHWFS_',num2str(nk*nrepeat),'.mat ISHWFS_3D;');
        eval(str);
        str = strcat('save DataSet\Zer_',num2str(nk*nrepeat),'.mat Zer;');
        eval(str);
        k = 0;
        clear ISHWFS_3D;
        clear Zer;
    end
end