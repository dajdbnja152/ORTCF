function [X_recov,S] = orhtc(M_0,Omega,rank,iter,lambda_1,lambda_2)
dim_ = size(M_0);
Y = zeros([[rank,rank],dim_(3:end)]);
X = zeros([[dim_(1),rank],dim_(3:end)]);
R = zeros([rank,dim_(2:end)]);
r = zeros([[rank,1],dim_(3:end)]);
O_ = zeros(dim_);
S = zeros(dim_);
L = randn([[dim_(1),rank],dim_(3:end)]);
L_old = zeros([[dim_(1),rank],dim_(3:end)]);
transformH.L = 'fft';transformH.rho = prod(dim_(3:end));
tol = 1e-7;

for i = 1: iter
    %fprintf("iter:%d\n",i);
    for n = 1: dim_(2)
        m = M_0(:,n,:,:);
        %更新右张量R和噪声张量S
        omega_ = Omega(:,n,:,:);
        [s,r,O] = R_update(m,r,L,omega_,rank,lambda_1,lambda_2,transformH);
        O_(:,n,:,:) = O;
        S(:,n,:,:) = s;
        R(:,n,:,:) = r;
        %更新左张量L
        Y = Y + H_tprod(r ,H_tran(r,transformH),transformH);
        X = X + H_tprod(m-s-O,H_tran(r,transformH),transformH);
        L = L_update(L,X,Y,lambda_1,transformH);
        %fprintf("diff_L:%f\n",norm(L(:)-L_old(:)));
        L_old = L;
    end
    X_recov = H_tprod(L,R,transformH);
end

function [s_,r_,O] = R_update(m_,r_,L_,Omega_,rank_,lambda_1,lambda_2,transform)
    d_ = size(m_);
    L_1 = H_tprod(H_tran(L_,transform),L_,transform)+lambda_1*H_teye([[rank_,rank_],d_(3:end)],transform);%identity([[rank_,rank_],d_(3:end)]);
    L_1_inv = H_tinv(L_1,transform);
    L_2 = H_tran(L_,transform);
    L_tilde = H_tprod(L_1_inv,L_2,transform);
    s_ = zeros(d_);
    s_new = s_;
    r_new = r_;
    converged = false;
    O = zeros(d_);
    iter_ = 0;
    while ~converged
        iter_ = iter_ + 1 ;
        s_mult = m_ - H_tprod(L_,r_,transform) - O;
        s_ = soft_threshold(s_mult,lambda_2);
        r_ = H_tprod(L_tilde,m_-s_-O,transform);
        O = m_ - s_ - H_tprod(L_,r_,transform);
        O = O .* (1-Omega_);
        error = max([norm(s_(:)-s_new(:),"fro"),norm(r_(:)-r_new(:),"fro")]);
        if error < tol
            converged = true;
        end
        if ~converged && iter_ >= 1000
            converged = true ;    
        end
        %fprintf("error:%d\n",error);
        s_new = s_;
        r_new = r_;
    end 
end

function [L_] = L_update(L_,X_,Y_,lambda_1,transform)
    d_ = size(X_);
    Y_ = Y_ + lambda_1 * H_teye(size(Y_),transform);%identity(size(Y_));
    for n_ = 3:length(d_)
        X_ = fft(X_,[],n_);
        Y_ = fft(Y_,[],n_);
        L_ = fft(L_,[],n_);
    end
    for j = 1:d_(2)
        for k = 1:prod(d_(3:end))
            L_(:,j,k) = L_(:,j,k) + (X_(:,j,k) - L_(:,:,k) * Y_(:,j,k))/Y_(j,j,k);
        end
    end
    for n_ = 3:length(d_)
        L_ = ifft(L_,[],n_);
    end
end

end