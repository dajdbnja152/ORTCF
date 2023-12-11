function [X_recov,S,time] = otrpca(M_0,rank,iter,lambda_1,lambda_2,transform)
dim_ = size(M_0);
Y = zeros([[rank,rank],dim_(3:end)]);
X = zeros([[dim_(1),rank],dim_(3:end)]);
R = zeros([rank,dim_(2:end)]);
r = zeros([[rank,1],dim_(3:end)]);
S = zeros(dim_);
L = randn([[dim_(1),rank],dim_(3:end)]);
L_old = zeros([[dim_(1),rank],dim_(3:end)]);
X_recov = tprod(L,R,transform);

for i = 1: iter
    for n = 1: dim_(2)
        tic;
        m = M_0(:,n,:,:);
        [s,r] = R_update(m,r,L,rank,lambda_1,lambda_2,transform);
        S(:,n,:,:) = s;
        R(:,n,:,:) = r;
        Y = Y + tprod(r ,tran(r,transform),transform);
        X = X + tprod(m-s,tran(r,transform),transform);
        L = L_update(L,X,Y,lambda_1,transform);
        L_old = L;
        time = toc;
    end
    X_recov = tprod(L,R,transform);
end

function [s_,r_] = R_update(m_,r_,L_,rank_,lambda_1,lambda_2,transform)
    d_ = size(m_);
    L_1 = tprod(tran(L_,transform),L_,transform)+lambda_1*teye(rank_,d_(3),transform);%identity([[rank_,rank_],d_(3:end)]);
    L_1_inv = tinv(L_1,transform);
    L_2 = tran(L_,transform);
    L_tilde = tprod(L_1_inv,L_2,transform);
    s_ = zeros(d_);
    s_new = s_;
    r_new = r_;
    error = 1;
    small_error = 0;
    tic;
    while small_error < 5
        r_ = tprod(L_tilde,m_-s_,transform);
        s_mult = m_ - tprod(L_,r_,transform);
        s_ = soft_threshold(s_mult,lambda_2);
        error = max(norm(s_(:)-s_new(:),"fro"),norm(r_(:)-r_new(:),"fro"));
        if error < 1e-4
             small_error = small_error + 1;
        end
        s_new = s_;
        r_new = r_;
        a = toc;
        if a > 30
            break
        end
    end 
end

function [L_] = L_update(L_,X_,Y_,lambda_1,transform)
    d_ = size(X_);
    d_y = size(Y_);
    Y_ = Y_ + lambda_1 * teye(d_y(2),d_y(3),transform);%identity(size(Y_));
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