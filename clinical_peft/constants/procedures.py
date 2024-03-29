PROCEDURES_MAP = {
    procedure_code: i
    for i, procedure_code in enumerate(
        [
            "340",
            "967",
            "389",
            "331",
            "34",
            "361",
            "396",
            "360",
            "372",
            "885",
            "991",
            "159",
            "439",
            "425",
            "441",
            "463",
            "456",
            "465",
            "451",
            "992",
            "394",
            "863",
            "887",
            "422",
            "131",
            "960",
            "457",
            "461",
            "387",
            "332",
            "966",
            "555",
            "399",
            "996",
            "392",
            "395",
            "884",
            "550",
            "501",
            "370",
            "349",
            "841",
            "549",
            "373",
            "378",
            "377",
            "518",
            "965",
            "939",
            "462",
            "541",
            "542",
            "374",
            "312",
            "431",
            "352",
            "413",
            "520",
            "234",
            "22",
            "527",
            "540",
            "388",
            "978",
            "314",
            "972",
            "512",
            "467",
            "345",
            "763",
            "341",
            "325",
            "810",
            "309",
            "577",
            "565",
            "311",
            "567",
            "921",
            "347",
            "815",
            "342",
            "776",
            "831",
            "808",
            "866",
            "384",
            "443",
            "838",
            "867",
            "784",
            "876",
            "786",
            "862",
            "452",
            "376",
            "860",
            "444",
            "896",
            "552",
            "573",
            "379",
            "963",
            "139",
            "353",
            "357",
            "381",
            "854",
            "857",
            "998",
            "995",
            "304",
            "505",
            "423",
            "454",
            "684",
            "656",
            "544",
            "659",
            "380",
            "545",
            "453",
            "464",
            "109",
            "151",
            "459",
            "471",
            "741",
            "753",
            "730",
            "688",
            "706",
            "566",
            "578",
            "547",
            "875",
            "393",
            "877",
            "800",
            "819",
            "778",
            "773",
            "536",
            "316",
            "317",
            "809",
            "598",
            "894",
            "329",
            "975",
            "805",
            "401",
            "371",
            "403",
            "796",
            "781",
            "793",
            "415",
            "437",
            "382",
            "886",
            "218",
            "275",
            "881",
            "470",
            "797",
            "319",
            "556",
            "865",
            "397",
            "946",
            "363",
            "528",
            "521",
            "920",
            "686",
            "590",
            "16",
            "206",
            "326",
            "334",
            "230",
            "767",
            "769",
            "997",
            "534",
            "339",
            "777",
            "922",
            "990",
            "821",
            "510",
            "779",
            "487",
            "571",
            "460",
            "790",
            "217",
            "964",
            "613",
            "442",
            "402",
            "114",
            "832",
            "872",
            "853",
            "525",
            "529",
            "833",
            "859",
            "212",
            "383",
            "499",
            "490",
            "324",
            "302",
            "717",
            "351",
            "824",
            "868",
            "851",
            "322",
            "834",
            "486",
            "702",
            "243",
            "113",
            "323",
            "132",
            "861",
            "434",
            "850",
            "535",
            "586",
            "153",
            "204",
            "546",
            "982",
            "584",
            "386",
            "485",
            "780",
            "785",
            "935",
            "202",
            "506",
            "554",
            "774",
            "405",
            "640",
            "954",
            "575",
            "492",
            "482",
            "961",
            "526",
            "973",
            "537",
            "168",
            "458",
            "390",
            "203",
            "231",
            "123",
            "970",
            "889",
            "446",
            "295",
            "369",
            "843",
            "609",
            "985",
            "509",
            "791",
            "695",
            "694",
            "663",
            "690",
            "531",
            "43",
            "516",
            "722",
            "124",
            "292",
            "519",
            "836",
            "118",
            "685",
            "239",
            "242",
            "424",
            "466",
            "890",
            "269",
            "974",
            "530",
            "469",
            "412",
            "683",
            "293",
            "579",
            "689",
            "435",
            "765",
            "806",
            "802",
            "807",
            "205",
            "359",
            "125",
            "468",
            "639",
            "483",
            "253",
            "766",
            "870",
            "893",
            "484",
            "709",
            "705",
            "391",
            "211",
            "165",
            "445",
            "438",
            "421",
            "560",
            "513",
            "543",
            "610",
            "828",
            "224",
            "210",
            "142",
            "511",
            "676",
            "355",
            "574",
            "450",
            "585",
            "839",
            "488",
            "976",
            "840",
            "601",
            "216",
            "801",
            "842",
            "222",
            "226",
            "320",
            "429",
            "515",
            "962",
            "503",
            "830",
            "803",
            "356",
            "404",
            "500",
            "597",
            "707",
            "502",
            "682",
            "652",
            "666",
            "481",
            "449",
            "428",
            "294",
            "934",
            "623",
            "625",
            "681",
            "522",
            "981",
            "315",
            "290",
            "251",
            "250",
            "200",
            "346",
            "570",
            "770",
            "64",
            "255",
            "115",
            "410",
            "479",
            "101",
            "818",
            "764",
            "321",
            "300",
            "13",
            "864",
            "343",
            "654",
            "658",
            "845",
            "816",
            "762",
            "102",
            "344",
            "980",
            "14",
            "814",
            "252",
            "539",
            "874",
            "184",
            "303",
            "11",
            "837",
            "274",
            "270",
            "263",
            "272",
            "419",
            "917",
            "559",
            "891",
            "611",
            "602",
            "771",
            "649",
            "348",
            "605",
            "813",
            "51",
            "923",
            "50",
            "895",
            "852",
            "792",
            "653",
            "514",
            "558",
            "599",
            "354",
            "942",
            "743",
            "660",
            "740",
            "699",
            "680",
            "756",
            "494",
            "333",
            "624",
            "971",
            "782",
            "12",
            "825",
            "823",
            "532",
            "221",
            "916",
            "638",
            "603",
            "241",
            "480",
            "15",
            "147",
            "163",
            "712",
            "244",
            "52",
            "54",
            "440",
            "858",
            "811",
            "385",
            "538",
            "220",
            "993",
            "430",
            "61",
            "63",
            "17",
            "55",
            "504",
            "517",
            "93",
            "563",
            "614",
            "497",
            "45",
            "798",
            "280",
            "994",
            "152",
            "944",
            "455",
            "358",
            "32",
            "213",
            "795",
            "227",
            "713",
            "999",
            "952",
            "207",
            "715",
            "524",
            "817",
            "704",
            "591",
            "10",
            "472",
            "145",
            "564",
            "187",
            "28",
            "164",
            "606",
            "39",
            "92",
            "626",
            "562",
            "493",
            "259",
            "262",
            "855",
            "950",
            "568",
            "91",
            "883",
            "287",
            "282",
            "804",
            "291",
            "122",
            "700",
            "977",
            "407",
            "820",
            "892",
            "869",
            "281",
            "62",
            "655",
            "633",
            "632",
            "182",
            "299",
            "827",
            "46",
            "576",
            "711",
            "734",
            "66",
            "40",
            "47",
            "129",
            "406",
            "400",
            "882",
            "41",
            "33",
            "48",
            "880",
            "42",
            "651",
            "180",
            "310",
            "822",
            "755",
            "489",
            "716",
            "735",
            "710",
            "826",
            "75",
            "31",
            "277",
            "721",
            "414",
            "736",
            "612",
            "160",
            "759",
            "856",
            "812",
            "491",
            "553",
            "847",
            "24",
            "931",
            "313",
            "433",
            "775",
            "44",
            "65",
            "932",
            "427",
            "273",
            "751",
            "754",
            "589",
            "783",
            "157",
            "225",
            "128",
            "134",
            "137",
            "127",
            "149",
            "106",
            "166",
            "72",
            "671",
            "110",
            "583",
            "53",
            "835",
            "261",
            "714",
            "296",
            "600",
            "642",
            "76",
            "229",
            "607",
            "121",
            "708",
            "337",
            "657",
            "701",
            "214",
            "692",
            "622",
            "631",
            "74",
            "126",
            "94",
            "644",
            "697",
            "643",
            "580",
            "673",
            "409",
            "794",
            "70",
            "604",
            "18",
            "533",
            "276",
            "641",
            "426",
            "73",
            "245",
            "279",
            "84",
            "829",
            "77",
            "144",
            "744",
            "80",
            "569",
            "21",
            "183",
            "761",
            "173",
            "71",
            "81",
            "930",
            "59",
            "174",
            "582",
            "844",
            "240",
            "215",
            "181",
            "362",
            "82",
            "758",
            "474",
            "750",
            "772",
            "233",
            "729",
            "436",
            "199",
            "83",
            "760",
            "398",
            "69",
            "551",
            "629",
            "630",
            "873",
            "849",
            "620",
            "350",
            "60",
            "175",
            "933",
            "23",
            "247",
            "237",
            "177",
            "914",
            "120",
            "260",
            "848",
            "143",
            "703",
            "411",
            "58",
            "693",
        ]
    )
}
