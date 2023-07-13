function [varargout]=HySime(varargin)
%
% HySime: Hyperspectral signal subspace estimation
%
%% -------------- HySime Sintax -------------- 
%
% [Ek,kf,]=HySime(y,HySime_Parameters);
%
%% --------------  Required inputs  -------------- 
%
% y - Observed hyperspectral dataset [L x N] matrix,
%     where L is the number of channels (bands),
%     N is the number of pixels (Lines x Columns).
%     Each column of y, a spectral vector, is a 
%     linear mixture of p endmembers signatures:
%
%              y = M*s + noise,
%
%     where M is an [L x p] matrix with endmembers 
%     signatures in each column, 
%     s is an [p x N] matrix with the endmembers 
%     abundance fractions, which are subject to 
%     non-negativity (s(i,j)>=0) and full-additivity 
%     ( sum(s(:,j)=1 ) constraints.
%
%% -------------- Optional inputs -------------- 
%
% HySime_Parameters - Structure with optional parameters
%                    (the structure can be given with all or some parameters)
% HySime_Parameters = struct('Noise',w, ...
%                            'Noise_Correlation_Matrix',Rw, ...
%                            'Percentage',1, ...
%                            'Display_figure','on', ...
%                            'Verbose','on');
% where:
%
% 'Noise'                    - noise or it estimates [L x N] matrix
%                              (optional parameter, if not given an 
%                              estimate is calculated.
%
% 'Noise_Correlation_Matrix' - noise correlation matrix [L x L] matrix
%                              (optional parameter, if not given an 
%                              estimate is calculated.
%
% 'Percentage'               - Option to return the eigenvectors that best
%                              represent a percentage of the spectral energy.
%                              optional parameter [scalar], between 0 and 1.
%
% 'Display_figure'           - Option to show figures while running HySime [string]
%                              'on' or 'off' default is 'on'
%
% 'Verbose'                  - Option to display information [string]
%                              'on' or 'off' default is 'on'
%
%% -------------- Outputs Parameters -------------- 
%
% kf  - signal subspace dimension
%
% Ek  - matrix which columns are the eigenvectors 
%       that span the signal subspace
%
%% -------------- Aditional Outputs -------------- 
% 
% Aditional output "signal_noise_estimates" struct 
% signal_noise_estimates = struct('Projection_Matrix_Percent',U,...
%                                 'Signal_Estimates',x,...
%                                 'Signal_Correlation_Matrix_Estimates',Rx,...   
%                                 'Noise_Estimates',n,...
%                                 'Noise_Correlation_Matrix_Estimates',Rn,...
%                                 'Mean Squared Error',cost_F,...
%                                 'Projection Error',Py,...
%                                 'Noise Power',Pn);
% where
%
% 'Projection_Matrix_Percent' - Matrix with the eigen vectors 
%                               that represent a percentage of the signal
% 'Noise_Estimates' - noise estimated by Hypernoise function
% 'Signal_Estimates' - signal estimates = observed data minus noise estimates
% 'Signal_Correlation_Matrix_Estimates' - signal correlation matrix
% 'Noise_Correlation_Matrix_Estimates' - noise correlation matrices 
% 'Mean Squared Error' - cost function which is a sum of
%                        'Projection Error' and 'Noise Power'
%
%
%% -------------- Brief HySime Description -------------- 
%
%  HySime - hyperspectral signal identification by minimum error
%  is a minimum mean  squared error based  approach to infer the 
%  signal subspace in hyperspectral imagery.
%  The method first  estimates  the signal and noise correlation
%  matrices  using  multiple  regression, and  then selects  the 
%  subset of eigenvectors of the  signal correlation matrix that
%  best represents  the  signal subspace  in  the  least squared
%  error sense.
%  The signal subspace is inferred by minimizing the sum  of the 
%  projection  error  power  with  the  noise  power, which  are, 
%  respectively, decreasing  and  increasing  functions  of  the
%  subspace dimension.  The overall scheme is fully unsupervised 
%  and it does not depend on any tuning parameters.
%
% More details in:
%
%  Jos¨¦ M. Bioucas-Dias and Jos¨¦ M. P. Nascimento,
%  "Hyperspectral Subspace Identification"
%  IEEE Transaction on Geoscience and Remote Sensing
%  vol 46, N.8, pp. 2435-2445, 2008.
%
%  For any comments contact the authors
%
%% -------------- Copyright -------------- 
%%
% Copyright (2005):        
% Jos¨¦ M.P. Nascimento (zen@isel.pt)
% & 
% Jos¨¦ Bioucas-Dias (bioucas@lx.it.pt)
%
% Created: January 2005
% Latest Revision: June 2011

%
% HySime is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and  distribute this 
% software for any purpose without fee is hereby granted,
% provided that  this entire  notice is included in all 
% copies of any software which is or includes a copy or 
% modification of this software and in all copies of the
% supporting documentation for such software.
% This  software is  being provided "as is", without any
% express or implied warranty. In particular, the authors
% do not make any representation or warranty of any kind 
% concerning the merchantability of this software or its 
% fitness for any particular purpose."
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% --------------  Read input parameters  -------------- 
%NARGINCHK
error(nargchk(1, 2, nargin))
[y,L,N,n,Rn,percent,vefig,verb] = read_hysime_params(varargin);
%
%% -------------- looking for output parameters -------------- 
%
if nargout > 3, error('Too many output parameters'); end;
if (nargout < 2) && verb, fprintf(1,'Warning: Output only the endmembers signatures.\n');end;

%% -------------- Estimate noise if not given -------------- 
if verb
    verbose='on';
else
    verbose='off';
end
if isnan(n)
  [Rn,n] = HyperNoise(y,'additive',verbose);
end
%% -------------- HySime method -------------- 
x = y - n;
if verb,fprintf(1,'Computing the correlation matrices\n');end
Ry = y*y'/N;   % sample correlation matrix 
Rx = x*x'/N;   % signal correlation matrix estimates 
if verb,fprintf(1,'Computing the eigen vectors of the signal correlation matrix\n');end
[E,D]=svd(Rx); % eigen values of Rx in decreasing order, equation (15)
dx = diag(D);

if verb,fprintf(1,'Estimating the number of endmembers\n');end
Rn=Rn+sum(diag(Rx))/L/10^10*eye(L);
Py = diag(E'*Ry*E);   %equation (23)
Pn = diag(E'*Rn*E);   %equation (24)
cost_F = -Py + 2* Pn; %equation (22)

kf = sum(cost_F<0);
[dummy,ind_asc] = sort( cost_F ,'ascend');
Ek = E(:,ind_asc(1:kf));
if verb,fprintf(1,'The signal subspace dimension is: k = %d\n',kf);end

% only for plot purposes, equation (19)
Py_sort =  trace(Ry) - cumsum(Py(ind_asc));
Pn_sort = 2*cumsum(Pn(ind_asc));
cost_F_sort = Py_sort + Pn_sort;

if vefig
   indice=1:50;
   figure
      set(gca,'FontSize',12,'FontName','times new roman')
      semilogy(indice,cost_F_sort(indice),'-',indice,Py_sort(indice),':',indice,Pn_sort(indice),'-.', 'Linewidth',2,'markersize',5)
      xlabel('k');ylabel('mse(k)');title('HySime')
      legend('Mean Squared Error','Projection Error','Noise Power')
end


varargout(1) = {kf};
if nargout >= 2, varargout(2) = {Ek};end
if nargout == 3
   if ~isnan(percent)
      %x0 = x-repmat(mean(x,2),[1 N]);
      %[Eo,Do]=svd(x0*x0'/N); 
      %dxo = diag(Do);
      %Px = dxo.*(dxo>0);
      Px = dx(ind_asc) .*(dx(ind_asc)>0);
      aux=find(cumsum(Px/sum(Px))>percent);
      if isempty(aux), aux=L; end;
      E_percent = E(:,ind_asc(1:aux(1)));
   else
      E_percent = E(:,ind_asc); 
   end
   
   signal_noise_estimates = struct('Projection_Matrix_Percent',E_percent,...
                                   'Signal_Estimates',x,...
                                   'Signal_Correlation_Matrix_Estimates',Rx,...   
                                   'Noise_Estimates',x,...
                                   'Noise_Correlation_Matrix_Estimates',Rx,...
                                   'Mean_Squared_Error',cost_F_sort,...
                                   'Projection_Error',Py_sort,...
                                   'Noise_Power',Pn_sort);
   varargout(3) = {signal_noise_estimates};
end
end %end of function [varargout]=HySime(varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INTERNAL FUNCTIONS
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% read_hysime_params - read all input parameters 
%                      and set default values if needed
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y,L,N,n,Rn,percent,vefig,verb] = read_hysime_params(varargin)

         % default parameters 
         n = nan;     % estimation is needed
         Rn = nan;    % estimation is needed
         percent = nan; 
         vefig = 1; 
         verb = 1;

         input_data = varargin{1}; % varargin is a cell with 2 cells inside
         n_input = numel(input_data);
         if n_input < 1
            error('Input observed data matrix is needed');
         end
         % reading required input matrix (observed data)
         Y = input_data{1};

         if isempty(Y) || ~isa(Y,'double') || any(isnan(Y(:))) || ndims(Y)~=2
            error('Wrong data matrix: Y should be an L (channels) x N (pixels) matrix, with double values.\n');
         end
         [L, N] = size(Y);  % L number of bands (channels)
                           % N number of pixels (Lines x Columns)
         if L > N 
            fprintf(1,'Warning: Aparently the number of observations is smaller than the number of channels. \n')
         end
         if n_input > 2
            fprintf(1,'Warning: Too many input parameters. Using default parameters instead. \n')
         end
         if (n_input == 1)
            fprintf(1,'Noise estimation required; Using default parameters. \n');
         end
         % checking for optional parameters struct
         if (n_input == 2)
            hysime_Parameters = input_data{2};
            if ~isstruct(hysime_Parameters)
               fprintf(1,'Warning: Invalid parameters struct. Using default parameters instead. \n');
            else
                % 'try' do not need 'catch', default parameters are already defined
                try % reading verbose first 
                    Verbose_in = lower(hysime_Parameters.Verbose); 
                    if ~any(strcmp(Verbose_in,{'on','off'}))
                       fprintf(1,'Warning: wrong Verbose option. Using default option. \n');
                    else verb = strcmp(Verbose_in,'on');
                    end
                end
                try 
                    n_in = hysime_Parameters.Noise;
                    [Ln, Nn]=size(n_in);
                    if any(isnan(n_in(:))) || ~isa(n_in,'double')
                       if verb,fprintf(1,'Wrong Noise matrix. \n');end
                    else
                       if L~=Ln || N~=Nn
                          if verb,fprintf(1,'Noise matrix is not consistent with data matrix. \n');end
                       else
                          n=n_in;
                       end
                    end
                end
                if ~isnan(n) % read noise correlation matrix only if noise option is valid
                  try % Assume that Noise is already verified!
                    R_in = hysime_Parameters.Noise_Correlation_Matrix;
                    [La, Lb]=size(R_in);
                    if any(isnan(R_in(:))) || ~isa(R_in,'double')|| L~=La || L~=Lb 
                       if verb,fprintf(1,'Wrong noise correlation matrix. \n');end
                       if ~isnan(n)
                          if verb,fprintf(1,'Computing noise correlation matrix. \n');end
                          Rn = diag(diag(n*n'/N));
                       end
                    else
                       Rn = R_in;
                    end
                  catch
                       Rn = diag(diag(n*n'/N));
                  end
                end
                try 
                    percent_in = hysime_Parameters.Percentage; 
                    if ~isscalar(percent_in) || ~isa(percent_in,'double') || percent_in<=0 || percent_in>1, 
                       if verb,fprintf(1,'Warning: percentage parameter should be a value between 0 and 1. Using default value: %d \n',percent);end
                    else
                        percent = percent_in;
                    end
                end
                try 
                    Ver_fig_in = lower(hysime_Parameters.Display_figure); 
                    if ~any(strcmp(Ver_fig_in,{'on','off'}))
                       if verb,fprintf(1,'Warning: wrong Display_figure option. Using default option. \n');end
                    else
                        vefig = strcmp(Ver_fig_in,'on');
                    end
                end
            end
         end
end % function read_hysime_params