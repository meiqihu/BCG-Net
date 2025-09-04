function [varargout]=HyperNoise(varargin)
%
%% -------------- estNoise Sintax -------------- 
%
% Syntax:
%         [Rw w] = HyperNoise(y);
%         [Rw w] = HyperNoise(y,noise_type,verbose);
%         [Rw] = HyperNoise(y,percent,noise_type,verbose);
%
%% -------------- Required inputs -------------- 
%
% y - Observed hyperspectral dataset [L x N] matrix,
%     where L is the number of channels (bands),
%     N is the number of pixels (Lines x Columns).
%     Each pixel is a mixture (linear or nonlinear)
%     of p endmembers signatures, i.e.,
%
%              y = x + n,
%
%     where x and n are the signal and noise matrices.
%
%% -------------- Optional inputs -------------- 
%
%    percent: [optional] Percentage of the number 
%             of pixels used to estimate the noise.
%             value between 0 and 1 (default=1)
%    noise_type: [optional] ('additive')|'poisson'
%    verbose: [optional] ('on')|'off'
%
%% -------------- Outputs Parameters -------------- 
%
%    Rw - noise correlation matrix estimates (LxL)
%    w  - matrix with the noise estimates for every pixel (LxN)
%         This matrix is returned when the input parameter percent is 1
%
%% -------------- Brief estNoise Description -------------- 
%
% HyperNoise: hyperspectral noise estimation.
% This function infers the noise in a 
% hyperspectral dataset, by assuming that the 
% reflectance at a given band is well modelled 
% by a linear regression on the remaining bands.
%
% More details in:
%
% Jos¨¦ M. Bioucas-Dias and Jos¨¦ M. P. Nascimento,
% "Hyperspectral Subspace Identification"
% IEEE Transaction on Geoscience and Remote Sensing
% vol 46, N.8, pp. 2435-2445, 2008.
%
%  For any comments contact the authors
%
%% -------------- Copyright -------------- 
%
% Copyright (2005):        
% Jos¨¦ M.P. Nascimento (zen@isel.pt)
% & 
% Jos¨¦ Bioucas-Dias (bioucas@lx.it.pt)
%
% Created: January 2005
% Latest Revision: June 2011

%
% DECA is distributed under the terms of
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

%% Default values
percent = 1;
percent_in = 0; % auxiliary variable that controls if the user gave the percent option
noise_type = 'additive'; 
verbose = 1; verb ='on'; 

%% Read input parameters
error(nargchk(1, 4, nargin))
if nargout > 2, error('too many output parameters'); end
y = varargin{1};
if isempty(y) || ~isa(y,'double') || any(isnan(y(:))) || ndims(y)~=2 
   error('Wrong dataset matrix: Y should be an L (channels) x N (pixels) matrix, with double values.\n');
end
for i=2:nargin 
   switch lower(varargin{i}) 
       case {'additive'}, noise_type = 'additive';
       case {'poisson'}, noise_type = 'poisson';
       case {'on'}, verbose = 1; verb = 'on';
       case {'off'}, verbose = 0; verb = 'off';
       otherwise 
           if isscalar(varargin{i}) && isreal(varargin{i}) && varargin{i}<1 && varargin{i}>0,
              percent = varargin{i};
              percent_in = 1;
              if nargout == 2
                 fprintf(1,'WARNING: to return the noise estimates, percent=1 must be used\n'); 
              end
           else    
              fprintf(1,'parameter [%d] not known.\n',i);
           end
   end
end

[L, N] = size(y);
if L<2, error('Too few bands to estimate the noise.'); end

if verbose
   fprintf(1,'Estimating %s noise based on %d pixels.\n',noise_type,round(percent*N)); 
end

if percent_in
   %percent = min([50000/N percent]); % 50e3 pixel should be enough to estimate the noise 
   idx = randperm(N);
   y = y(:,idx(1:round(percent*N)));
   [L, N] = size(y);
end

if strcmp(noise_type,'poisson')
       sqy = sqrt(y.*(y>0));                % prevent negative values
       [u ,Ru] = estAdditiveNoise(sqy,verb); % noise estimates
       x = (sqy - u).^2;                    % signal estimates 
       w = sqrt(x).*u*2;
       Rw = w*w'/N; 
else % additive noise
       [w, Rw] = estAdditiveNoise(y,verb);   % noise estimates        
end

varargout(1) = {Rw};
if percent<1, w=[];end
if nargout == 2, varargout(2) = {w}; end
end % end of function [varargout]=HyperNoise(varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Internal Function - Estimates the noise by linear regression
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w,Rw]=estAdditiveNoise(r,verbose)

small = 1e-6;
verbose = ~strcmp(lower(verbose),'off');
[L ,N] = size(r);
% the noise estimation algorithm
w=zeros(L,N);
if verbose 
   fprintf(1,'computing the sample correlation matrix and its inverse\n');
end
RR=r*r';                  % equation (11)
RRi=inv(RR+small*eye(L)); % equation (11)
if verbose, fprintf(1,'computing band    ');end;
for i=1:L
    if verbose, fprintf(1,'\b\b\b%3d',i);end;
    % equation (14)
    XX = RRi - (RRi(:,i)*RRi(i,:))/RRi(i,i);
    RRa = RR(:,i); RRa(i)=0; % this remove the effects of XX(:,i)
    % equation (9)
    beta = XX * RRa; beta(i)=0; % this remove the effects of XX(i,:)
    % equation (10)
    w(i,:) = r(i,:) - beta'*r; % note that beta(i)=0 => beta(i)*r(i,:)=0
end
if verbose, fprintf(1,'\ncomputing noise correlation matrix\n');end
Rw=diag(diag(w*w'/N));
end % end of function [w,Rw]=estAdditiveNoise(r,verbose);