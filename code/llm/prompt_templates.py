"""
File for prompt templates
"""

system_prompt = (
        "Du bist ein Deutschlehrer der 9. Klasse und analysierst Aufsätze deiner Schüler. "
        "Du überprüfst dabei, ob die Aufsätze alle zu erwartenden Bestandteile enthält."
    )

system_prompt_article = (
        "Du bist ein Deutschlehrer der 9. Klasse und analysierst Aufsätze deiner Schüler. "
        "Du überprüfst dabei, ob die Aufsätze alle zu erwartenden Bestandteile enthält."
        "\nHier ist der Lesetext, auf den der Aufsatz Bezug nimmt.\n{}"
    )

user_prompt_background = (
        "Hier liegt ein argumentativer Aufsatz vor, den ein Schüler mit Bezug auf {} "
        "Lesetext geschrieben hat. "
        "Der Aufsatz wurde in Sätzen unterteilt. Die Sätze sind nummeriert. Das Format dabei is 'Satznummer: Satz'.\n"
        "Jeder Satz wird einer der folgenden 5 Funktionen zugeordnet:\n'_Einleitung' bedeutet: "
        "Der Satz leitet den Aufsatz ein und gibt evtl. Informationen zum Lesetext und zum Thema des Aufsatzes.\n"
        "'_Pro_aus_Lesetext' bedeutet: Der Satz gibt ein Pro-Argument aus dem Lesetext wider.\n"
        "'_Con_aus_Lesetext' bedeutet: Der Satz gibt ein Kontra-Argument aus dem Lesetext wider.\n"
        "'_Eigen' bedeutet: Der Satz gibt eigene Meinungen und Argumente des Schülers wider.\n"
        "'_Sonstiges' bedeutet: Der Satz kann keiner der vorherigen 4 Funktionen zugeordnet werden.\n"
    )
user_prompt_command = (
    "Werte den Schüleraufsatz aus und ordne jeden Satz einer der genannten 5 Funktionen zu. Das Format des Outputs "
    "soll ausschließlich wie folgt sein: 'Satznummer: Funktion', zum Beispiel '1: _Einleitung' oder '8: _Eigen'. \n"
    "Schüleraufsatz\n\n{}"
)