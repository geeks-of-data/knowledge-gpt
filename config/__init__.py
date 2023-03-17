environment = "dev"

if environment == "dev":
    from .dev import *
else:
    from .prod import *
