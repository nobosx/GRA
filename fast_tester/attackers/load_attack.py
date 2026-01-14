from attackers.RayS_attacker import RaySAttacker
from attackers.GuidedRay_attacker import GuidedRayAttacker
from attackers.HSJA_attacker import HSJAAttacker
from attackers.bounce_attacker import BounceAttacker
from attackers.Tangent_attacker import TangentAttacker
from attackers.SignOPTLinf_attacker import SignOPTLinfAttacker
from attackers.RayST_attacker import RaySTAttacker
from attackers.powerbounce_attacker import PowerBounceAttacker
from attackers.fasttester_attacker import FastTesterAttacker
from config import SUPPORTED_ATTACK_METHOD

def build_concrete_attacker(victim_model, attack_config: dict):
    """
    Build an adversarial attacker by attacker method.

    Args:
        victim_model (VictimModel): The victim model.
        attack_config (dict): Attack configure. The entries 'method', 'norm', 'targeted', 'max_query_count', 'early_stop', 'verbose' must be contained. If the entry 'early_stop' is True, 'epsilon' must be contained.
    
    Returns:
        attacker (BasicAttacker): The constructed attacker.
    """
    for config_entry in ['method', 'norm', 'targeted', 'max_query_count', 'early_stop', 'verbose']:
        assert config_entry in attack_config, f"Basic attack parameter {config_entry} must be provided in the attack_config!"
    assert attack_config['method'] in SUPPORTED_ATTACK_METHOD, f"Attack method {attack_config['method']} is not supported!"
    if attack_config['early_stop']:
        assert 'epsilon' in attack_config, "Early stop is enabled, thus epsilon msut be provided in the attack_config!"
    else:
        attack_config['epsilon'] = None
    
    # Build concrete attcker by attack_method
    attack_args = {
        'model': victim_model,
        'norm_order': attack_config['norm'],
        'targeted': attack_config['targeted'],
        'max_query_count': attack_config['max_query_count'],
        'early_stop': attack_config['early_stop'],
        'epsilon_metric': attack_config['epsilon'],
        'verbose': attack_config['verbose'],
    }
    # Note: Registration for a new attacker is required here!
    if attack_config['method'] == 'RayS':
        attacker = RaySAttacker(**attack_args)
    elif attack_config['method'] == 'GuidedRay':
        if 'augment_method' in attack_config:
            attack_args['augment_method'] = attack_config['augment_method']
        attacker = GuidedRayAttacker(**attack_args)
    elif attack_config['method'] == 'fast_test':
        if 'augment_method' in attack_config:
            attack_args['augment_method'] = attack_config['augment_method']
        attacker = FastTesterAttacker(**attack_args)
    elif attack_config['method'] == 'HSJA':
        attacker = HSJAAttacker(**attack_args)
    elif attack_config['method'] == 'bounce':
        attacker = BounceAttacker(**attack_args)
    elif attack_config['method'] == 'Tangent':
        attacker = TangentAttacker(**attack_args)
    elif attack_config['method'] == 'Sign_OPT_Linf':
        attacker = SignOPTLinfAttacker(**attack_args)
    elif attack_config['method'] == 'RayST':
        attacker = RaySTAttacker(**attack_args)
    elif attack_config['method'] == 'power_bounce':
        attack_args['augment_method'] = attack_config['augment_method']
        attack_args['T'] = attack_config['T']
        attacker = PowerBounceAttacker(**attack_args)
    return attacker