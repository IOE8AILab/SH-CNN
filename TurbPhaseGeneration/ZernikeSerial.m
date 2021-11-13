%% this function is used to generate the Zernike coefficients that satisfy the turbulence parameters
%% D: diameter of the telescope; r0: Fried parameter; 
%% nZernike: number of Zernike modes;L: the number of frames to be generated
%% 2021-08-12, Youming Guo, Institue of Optics and Electronics, Chinese Academy of Sciences.
function Zer = ZernikeSerial(D,r0,nZernike,L)
    Cz = ZernikeCovarianceMat(nZernike).*(D/r0).^(5/3);
    [U,S,V] = svd(Cz);
    KL = randn(nZernike,L);
    Zer = U*sqrt(S)*KL;
    Zer = Zer';
return