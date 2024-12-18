# Definindo códigos de cores e estilos
class TerminalStyles:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    COLORS = {
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        'PURPLE': '\033[95m'
    }


class UserInput:
    @staticmethod
    def select_model():
        models = {
            '1': ('AFM Only', 'unet_afm_1_channels_only_AFM_CosHeightSum'),
            '2': ('YOLO-AFM', 'unet_afm_2_channels_like_yolo_opt_afm'),
            '3': ('Optical-Only', 'unet_afm_2_channels_only_optical')
        }
        print(f'''{TerminalStyles.BOLD}Select your model:{TerminalStyles.RESET}
                 1 - {TerminalStyles.COLORS['PURPLE']}{models['1'][0]}{TerminalStyles.RESET}
                 2 - {TerminalStyles.COLORS['GREEN']}{models['2'][0]}{TerminalStyles.RESET}
                 3 - {TerminalStyles.COLORS['BLUE']}{models['3'][0]}{TerminalStyles.RESET}
                 ____________________________________''')
        
        user_input = input()
        selected_model = models.get(user_input)
        
        if selected_model:
            print(f'\nModel {TerminalStyles.BOLD}{TerminalStyles.COLORS["CYAN"]}__{selected_model[0]}__{TerminalStyles.RESET} selected\n')
            return selected_model[1]
        
        else:
            print(f'{TerminalStyles.COLORS["RED"]}Invalid selection. Please choose a valid option.{TerminalStyles.RESET}')
            return None

    @staticmethod
    def get_user_confirmation(prompt):
        while True:
            response = input(f'{TerminalStyles.BOLD}{prompt}{TerminalStyles.RESET} ({TerminalStyles.BOLD}[yes]{TerminalStyles.RESET}/no): ').strip().lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                return True
