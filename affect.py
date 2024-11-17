predefined_actions = [
    {
        "id": 0,
        "name": "reject command",
        "activities": [
            "EAC0018"
        ],
    },
    {
        "id": 1,
        "name": "accept command",
        "activities": [
            "EAC0018"
        ],
    },
    {
        "id": 2,
        "name": "reject command/(Domain Controller)Cleanup sid History",
        "activities": [
            "EAC0018",
            "EAC0014"
        ],
    },
    {
        "id": 3,
        "name": "accept command/(Domain Controller)Cleanup sid History",
        "activities": [
            "EAC0018",
            "EAC0014"
        ],
    },
        {
        "id": 4,
        "name": "reject command/(Pfsense)reject L4 samba(tcp 445) ticket(between 192.168.10.10 and 192.168.10.30)",
        "activities": [
            "EAC0016"
        ],
    },
    {
        "id": 5,
        "name": "accept command/(Pfsense)reject L4 samba(tcp 445) ticket(between 192.168.10.10 and 192.168.10.30)",
        "activities": [
            "EAC0016"
        ],
    },
    {
        "id": 6,
        "name": "reject command/(Pfsense)reject L4 kerboros(tcp 445) ticket(between 192.168.10.10 and 192.168.10.30)",
        "activities": [
            "EAC0016"
        ],
    },
    {
        "id": 7,
        "name": "accept command/(Pfsense)reject L4 kerboros ticket(between 192.168.10.10 and 192.168.10.30)",
        "activities": [
            "EAC0016"
        ],
    },
]