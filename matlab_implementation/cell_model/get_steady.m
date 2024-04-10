%% get_steady.m
% Function for getting the system's steady state
% returns the steady state x and time t taken to achieve it

%%

function last_x=get_steady(sim,... % simulator object
    Delta,... % error tolerance (simulation stops when sum of square errors between current and previous state is less than Delta)
    Max_iter) % maximum number of iterations over which the cycle is run

    % iterate integrations over sim.tf hours until steady state is reached
    last_x=zeros(1,size(sim.init_conditions,1)); % assume that at iteration zero, all variables are at 0
    for i=1:Max_iter
        % integrate
        sim = sim.simulate_model;
        
        if(norm(sim.x(end,:)-last_x)<Delta) % if SS reached, quit
            last_x=sim.x(end,:);
            break
        else % if not, continue integration, making the current state the new initial condition
%             if(i==Max_iter)
%                 disp('Warning! SS not reached yet')
%             end
            last_x=sim.x(end,:);
            % mRNAs
            sim.init_conditions('m_a')=last_x(1);
            sim.init_conditions('m_r')=last_x(2);
            % proteins
            sim.init_conditions('p_a')=last_x(3);
            sim.init_conditions('R')=last_x(4);
            % tRNAs
            sim.init_conditions('tc')=last_x(5);
            sim.init_conditions('tu')=last_x(6);
            % inactivated ribosomes
            sim.init_conditions('Bcm')=last_x(7);
            % nutrient quality and chloramphenicol
            sim.init_conditions('s')=last_x(8);
            sim.init_conditions('h')=last_x(9);

            % heterologous
            x_het=last_x(10 : (9+2*sim.num_het) );
            for j=1:sim.num_het
                % mRNA
                sim.init_conditions(['m_',sim.het.names{j}])=x_het(j);
                % protein
                sim.init_conditions(['p_',sim.het.names{j}])=x_het(sim.num_het+j);
            end
            
            % record the ss and time taken to reach it into output
            last_x=sim.x(end,:);
        end
    end
end