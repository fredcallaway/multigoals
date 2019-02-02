require('lme4')
require('nlme')
library(lmerTest)

df = read.csv('/Users/carlos/pu/multigoals/analysis/geom_k_models.csv')
df$experiment = ifelse(df$incentive %in% c('1x', '3x'), 'exp2', 'exp1')

m = lme(geom_k_p ~ factor(incentive), random = ~1|pid, data=df[df$experiment=='exp2',], method='ML')
anova(m)
summary(m)

m = lme(geom_exp_k ~ factor(incentive), random = ~1|pid, data=df[df$experiment=='exp2',], method='ML')
anova(m)
summary(m)

m = lm(geom_k_p ~ factor(experiment), data=df)
anova(m)

m = lm(geom_exp_k ~ factor(experiment), data=df)
anova(m)

# now trying paired test...
wide_exp2 = reshape(df[df$experiment=='exp2',], idvar='pid', timevar='incentive', direction='wide')
t.test(wide_exp2$geom_k_p.1x, wide_exp2$geom_k_p.3x, paired=TRUE)


### Trying for 1.4

df = read.csv('/Users/carlos/pu/multigoals/analysis/geom_k_models.csv')
df$experiment = ifelse(df$condition %in% c('No Bonus', 'Bonus'), 'exp4', 'exp1')

m = lme(geom_k_p ~ factor(condition), random = ~1|pid, data=df[df$experiment=='exp4',], method='ML')
anova(m)
summary(m)

m = lme(geom_exp_k ~ factor(condition), random = ~1|pid, data=df[df$experiment=='exp4',], method='ML')
anova(m)
summary(m)

# now trying paired test...
wide_exp4 = reshape(df[df$experiment=='exp4',], idvar='pid', timevar='condition', direction='wide')

t.test(wide_exp4$`geom_k_p.Bonus`, wide_exp4$`geom_k_p.No Bonus`, paired=TRUE, alternative='less')
t.test(wide_exp4$`geom_exp_k.Bonus`, wide_exp4$`geom_exp_k.No Bonus`, paired=TRUE, alternative='greater')

wilcox.test(wide_exp4$`geom_k_p.Bonus`, wide_exp4$`geom_k_p.No Bonus`, paired=TRUE, alternative='less')
wilcox.test(wide_exp4$`geom_exp_k.Bonus`, wide_exp4$`geom_exp_k.No Bonus`, paired=TRUE, alternative='greater')

###

df = read.csv('/Users/carlos/pu/multigoals/analysis/comparing-num-actions.csv')

onlyexp3 = df[df$high_stakes %in% c('True', 'False'),]
#m = lmer(num_actions ~ factor(high_stakes) + (1|pid), data=onlyexp3, REML=FALSE)
#anova(m)

m = lmer(num_actions ~ factor(no_bonus) + (1|pid) + (1|initial_state), data=onlyexp3, REML=FALSE)
anova(m)

m = lmer(num_actions ~ factor(trial_repeat) + (1|pid) + (1|initial_state), data=onlyexp3, REML=FALSE)
anova(m)


comparing_experiments = df[]
comparing_experiments$experiment = ifelse(comparing_experiments$high_stakes == '1.1', 'exp1', 'exp2')
#m = lme(num_actions ~ factor(experiment), random = ~1|pid, data=comparing_experiments, method='ML')

m = lm(num_actions ~ factor(experiment), data=comparing_experiments)
anova(m)
