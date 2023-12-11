% Demo for online robust tensor recovery in 10% noise (table1 in paper) 
clear all;
addpath(genpath(pwd))
rmpath(genpath('High-order tproduct toolbox'))
load('PeMS08_10.mat')
[n1,d,n2] = size(T);
transform.L = @fft; transform.l = n2; transform.inverseL = @ifft;
window_size = 5;
for i = 1:d
    day = T(:,i,:);
    day = reshape(day, [n1, n2]);
    for row = 1:n1
        for col = 1:n2
            left = max(1, row - floor(window_size/2));
            right = min(n1, row + floor(window_size/2));
            day_smooth(row, col) = mean(day(left:right,col));
        end
    end
    T_smooth(:,i,:) = day_smooth;
end
T = T_smooth;
ten = T(:,32:62,:);
[n1,d,n2] = size(ten);
for iter = 1:10
rng(iter);
X_0 = ten;
idx = find(ten);
p_tra = 0.8;
p_tst = 0.1;
rand_idx = randperm(length(idx));
traIdx = rand_idx(1:floor(length(idx)*p_tra));
tstIdx = rand_idx(floor(length(idx)*p_tra)+1:floor(length(idx)*(p_tra+p_tst)));
valIdx = rand_idx(floor(length(idx)*(p_tra+p_tst))+1: end);
Omega_tra = zeros(n1,d,n2);
Omega_tst = zeros(n1,d,n2);
Omega_val = zeros(n1,d,n2);
Omega_tra(idx(traIdx)) = 1;
Omega_tst(idx(tstIdx)) = 1;
Omega_val(idx(valIdx)) = 1;

p_s = 0.1; % 噪声
var = 200;
S_0 = rand(n1,d,n2); 
S_0(S_0<p_s) = 1;   
S_0(S_0~=1) = 0;
S_0 = S_0 .* Omega_tra;
C_0 = S_0 .*var;
M_0 = C_0 + X_0 .* Omega_tra;
rank = 5;
lambda1 = [500,1000,1500,2000];
lambda3 = [1e+0,1e+1,1e+2,1e+3];

fprintf("============ortcf=================\n")
[X_recov1,RSE1(iter),RMSE1(iter),R1(iter),time1(iter)] = run_ortcf(T,X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda1,lambda3,transform);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE1(iter),RMSE1(iter),R1(iter),time1(iter));

fprintf("=============Ormcf================\n")
[X_recov2,RSE2(iter),RMSE2(iter),R2(iter),time2(iter)] = run_ormcf(T,X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda1,lambda3);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE2(iter),RMSE2(iter),R2(iter),time2(iter));

fprintf("===============ortc===============\n");
[X_recov3,RSE3(iter),RMSE3(iter),R3(iter),time3(iter)] = run_ortc(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda1,transform);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE3(iter),RMSE3(iter),R3(iter),time3(iter));

fprintf("=============lrtcR_tnn================\n")
[X_recov4,RSE4(iter),RMSE4(iter),R4(iter),time4(iter)] = run_tnn(X_0,C_0,Omega_tra,Omega_val,Omega_tst,lambda1);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE4(iter),RMSE4(iter),R4(iter),time4(iter));
 
fprintf("=============OLRTR================\n")
[X_recov5,RSE5(iter),RMSE5(iter),R5(iter),time5(iter)] = run_olrtr(X_0,C_0,Omega_tra,Omega_val,Omega_tst,lambda3);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE5(iter),RMSE5(iter),R5(iter),time5(iter));

fprintf("=============OSTD================\n")
[X_recov6,RSE6(iter),RMSE6(iter),R6(iter),time6(iter)] = run_OSTD(X_0,C_0,Omega_tra,Omega_val,Omega_tst,lambda3);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE6(iter),RMSE6(iter),R6(iter),time6(iter));

fprintf("=============OLSTEC================\n")
[X_recov7,RSE7(iter),RMSE7(iter),R7(iter),time7(iter)] = run_olstec(X_0,C_0,Omega_tra,Omega_val,Omega_tst);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE7(iter),RMSE7(iter),R7(iter),time7(iter));

fprintf("=============GRASTA================\n")
[X_recov8,RSE8(iter),RMSE8(iter),R8(iter),time8(iter)] = run_grasta(X_0,C_0,Omega_tra,Omega_val,Omega_tst);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE8(iter),RMSE8(iter),R8(iter),time8(iter));

fprintf("=============ORPCA================\n")
[X_recov9,RSE9(iter),RMSE9(iter),R9(iter),time9(iter)] = run_orpca(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda1);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE9(iter),RMSE9(iter),R9(iter),time9(iter));

fprintf("=============OTRPCA================\n")
[X_recov10,RSE10(iter),RMSE10(iter),R10(iter),time10(iter)] = run_otrpca(X_0,C_0,Omega_tra,Omega_val,Omega_tst,rank,lambda1,transform);
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",RSE10(iter),RMSE10(iter),R10(iter),time10(iter));

end

fprintf("\n")
fprintf("=====================final result=====================\n");
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE1),mean(RMSE1),mean(R1),mean(time1));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE2),mean(RMSE2),mean(R2),mean(time2));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE3),mean(RMSE3),mean(R3),mean(time3));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE4),mean(RMSE4),mean(R4),mean(time4));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE5),mean(RMSE5),mean(R5),mean(time5));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE6),mean(RMSE6),mean(R6),mean(time6));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE7),mean(RMSE7),mean(R7),mean(time7));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE8),mean(RMSE8),mean(R8),mean(time8));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE9),mean(RMSE9),mean(R9),mean(time9));
fprintf("RSE:%.5f\tRMSE:%.5f\tR2:%.5f\ttime:%.5f\n",mean(RSE10),mean(RMSE10),mean(R10),mean(time10));