%% this function is used to generate the covariance matrix of Zernike modes
%% up to nZernike
function Cz = ZernikeCovarianceMat(nZernike)
    n_mode = nZernike;
    Cz = zeros(n_mode,n_mode);
    for i = 1:n_mode
        [ni,mi] = nmzern(i+1);
        for j = 1:n_mode
            [nj,mj] = nmzern(j+1);
            if mi == mj
                m = mi;
                if (m == 0) | (m ~= 0 && mod(i - j,2) == 0)
                    Cz(i,j) = (gamma(14/3)*(4.8*gamma(1.2))^(5/6)*(gamma(11/6))^2/(2^(8/3)*pi))*(-1)^((ni+nj-2*m)/2)*((ni+1)*(nj+1))^(1/2)*gamma((ni+nj-5/3)/2)/...
                        (gamma((ni-nj+17/3)/2)*gamma((nj-ni+17/3)/2)*gamma((nj+ni+23/3)/2));
                end
            end
        end
    end
end