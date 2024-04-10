%% cell_formulae.m
% collection of formulae for rate and activation functions

%%
classdef cell_formulae
   
    methods (Access = public)
        % apparent risbome dissociation constants
        function k=k(obj,epsilon,kplus,kminus,n,kcmh) 
            k=(kminus+epsilon./n+kcmh)./kplus;
        end

        % growth rate
        function l = l(obj,par,epsilon,B) 
            l=epsilon.*B./par('M');
        end

        % ribosomal gene transcription regulation
        function F_r = F_r(obj,par,T) 
            F_r=T./(T+par('tau'));
        end
        
        % translation elongation rate
        function e = e(obj,par,tc)
            e = par('e_max')*tc./(tc+par('K_e'));
        end
        
        % tRNA charging rate
        function nu = nu(obj,par,tu,s) 
            nu = par('nu_max').*(tu./(tu+par('K_nut'))).*s;
        end
        
        % tRNA synthesis rate
        function psi = psi(obj,par,T)
            psi = par('psi_max').*T./(T+par('tau'));
        end
        
    end

end