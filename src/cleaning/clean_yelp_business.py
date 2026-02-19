"""
Yelp Business Data Cleaning Script
Phase 2: Data Cleaning for Yelp Dataset

This script implements all required cleaning tasks:
1. Standardize column names
2. Clean location fields (lat/lon, ZIP codes)
3. Deduplicate records
4. Handle missing values
5. Normalize business categories using classical methods
"""

import pandas as pd
import re
from pathlib import Path
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings('ignore')


class YelpDataCleaner:
    """Clean and normalize Yelp business data"""

    def __init__(self, input_file):
        """
        Initialize the cleaner

        Parameters:
        -----------
        input_file : str or Path
            Path to raw Yelp data CSV
        """
        self.input_file = Path(input_file)
        self.df = None
        self.cleaning_stats = {}

    def load_data(self):
        """Load the raw Yelp data"""
        print("Loading Yelp data...")
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df)} businesses")
        self.cleaning_stats['original_count'] = len(self.df)
        return self

    def standardize_column_names(self):
        """
        Task 1: Standardize column names to snake_case

        Converts all column names to lowercase with underscores
        """
        print("\n--- Standardizing Column Names ---")

        new_cols = {}
        for col in self.df.columns:
            new_col = col.lower().strip()
            new_col = re.sub(r'[^\w\s]', '', new_col)
            new_col = re.sub(r'\s+', '_', new_col)
            new_cols[col] = new_col

        self.df.rename(columns=new_cols, inplace=True)

        print(f"Standardized {len(new_cols)} column names")
        print(f"Columns: {list(self.df.columns)}")
        self.cleaning_stats['columns'] = list(self.df.columns)
        return self

    def clean_location_fields(self):
        """
        Task 2: Clean location fields (latitude, longitude, ZIP codes)

        - Validate lat/lon within valid ranges and Philadelphia area
        - Standardize ZIP codes to 5-digit format
        - Flag and handle invalid coordinates
        """
        print("\n--- Cleaning Location Fields ---")

        # Philadelphia area bounds (approximate)
        PHILLY_LAT_MIN, PHILLY_LAT_MAX = 39.5, 40.5
        PHILLY_LON_MIN, PHILLY_LON_MAX = -76.0, -74.5

        # Validate latitude
        invalid_lat = (
            (self.df['latitude'] < -90) |
            (self.df['latitude'] > 90) |
            self.df['latitude'].isna()
        )
        print(f"Invalid latitude values: {invalid_lat.sum()}")

        # Validate longitude
        invalid_lon = (
            (self.df['longitude'] < -180) |
            (self.df['longitude'] > 180) |
            self.df['longitude'].isna()
        )
        print(f"Invalid longitude values: {invalid_lon.sum()}")

        # Check if coordinates are within Philadelphia area
        outside_philly = (
            (self.df['latitude'] < PHILLY_LAT_MIN) |
            (self.df['latitude'] > PHILLY_LAT_MAX) |
            (self.df['longitude'] < PHILLY_LON_MIN) |
            (self.df['longitude'] > PHILLY_LON_MAX)
        )
        print(f"Coordinates outside Philadelphia area: {outside_philly.sum()}")

        # Create flag for valid coordinates
        self.df['valid_coordinates'] = ~(invalid_lat | invalid_lon | outside_philly)
        print(f"Businesses with valid coordinates: {self.df['valid_coordinates'].sum()}")

        # Clean postal codes - standardize to 5-digit format
        def clean_zipcode(zipcode):
            """Extract 5-digit ZIP code"""
            if pd.isna(zipcode):
                return None
            zipcode = str(zipcode).strip()
            digits = re.sub(r'\D', '', zipcode)
            if len(digits) >= 5:
                return digits[:5]
            return None

        self.df['postal_code'] = self.df['postal_code'].apply(clean_zipcode)
        missing_zip = self.df['postal_code'].isna().sum()
        print(f"Missing ZIP codes after cleaning: {missing_zip}")

        self.cleaning_stats['invalid_coordinates'] = invalid_lat.sum() + invalid_lon.sum()
        self.cleaning_stats['missing_zip'] = missing_zip
        return self

    def handle_missing_values(self):
        """
        Task 3: Handle missing values with documented strategy

        Strategy:
        - Drop businesses with invalid/missing coordinates (critical for geospatial matching)
        - Keep businesses with missing categories (will mark as 'Uncategorized')
        - Keep businesses with missing hours/attributes (not critical for integration)
        """
        print("\n--- Handling Missing Values ---")

        # Report missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        print("\nMissing value percentages:")
        for col in self.df.columns:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)")

        # Strategy 1: Drop businesses with invalid coordinates
        before = len(self.df)
        self.df = self.df[self.df['valid_coordinates'] == True].copy()
        dropped_coords = before - len(self.df)
        print(f"\nDropped {dropped_coords} businesses with invalid coordinates")

        # Strategy 2: Keep missing categories but mark them
        missing_cats = self.df['categories'].isna().sum()
        if missing_cats > 0:
            print(f"Keeping {missing_cats} businesses with missing categories (will mark as 'Uncategorized')")
            self.df['categories'] = self.df['categories'].fillna('Uncategorized')

        # Strategy 3: Keep missing hours and attributes (not critical)

        self.cleaning_stats['dropped_invalid_coords'] = dropped_coords
        self.cleaning_stats['count_after_missing_values'] = len(self.df)
        return self

    def deduplicate_records(self):
        """
        Task 4: Deduplicate records

        Deduplication rules:
        1. Remove exact duplicates based on business_id (primary key)
        2. Check for fuzzy duplicates: same name + similar address
        3. Keep the record with more reviews if duplicates found
        """
        print("\n--- Deduplicating Records ---")

        initial_count = len(self.df)

        # Rule 1: Remove exact business_id duplicates
        business_id_dupes = self.df.duplicated(subset=['business_id'], keep=False).sum()
        print(f"Business_id duplicates found: {business_id_dupes}")
        self.df = self.df.drop_duplicates(subset=['business_id'], keep='first')
        print(f"After removing business_id duplicates: {len(self.df)} businesses")

        # Rule 2: Fuzzy duplicates (same name + similar address)
        print("\nChecking for fuzzy duplicates (same name + similar address)...")
        fuzzy_dupes = 0

        # Sort by review_count to keep higher-reviewed businesses
        self.df = self.df.sort_values('review_count', ascending=False)
        self.df['name_normalized'] = self.df['name'].str.lower().str.strip()

        to_drop = []
        for _, group in self.df.groupby('name_normalized'):
            if len(group) > 1:
                addresses = group['address'].fillna('').tolist()
                indices = group.index.tolist()

                for i in range(len(addresses)):
                    for j in range(i + 1, len(addresses)):
                        similarity = fuzz.ratio(addresses[i].lower(), addresses[j].lower())
                        if similarity > 80:
                            to_drop.append(indices[j])
                            fuzzy_dupes += 1

        if to_drop:
            self.df = self.df.drop(to_drop)
            print(f"Removed {fuzzy_dupes} fuzzy duplicates")

        self.df = self.df.drop(columns=['name_normalized'])

        duplicates_removed = initial_count - len(self.df)
        print(f"\nTotal duplicates removed: {duplicates_removed}")
        print(f"Businesses remaining: {len(self.df)}")

        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        self.cleaning_stats['count_after_deduplication'] = len(self.df)
        return self

    def normalize_categories(self):
        """
        Task 5: Normalize Yelp business categories using classical methods

        Strategy:
        1. Split comma-separated categories into individual categories
        2. Clean and normalize category names (lowercase, strip whitespace)
        3. Group similar categories using string matching and taxonomy
        4. Map to standardized category taxonomy
        """
        print("\n--- Normalizing Business Categories ---")

        # Step 1: Split categories
        def split_categories(cat_string):
            if pd.isna(cat_string) or cat_string == 'Uncategorized':
                return ['Uncategorized']
            return [c.strip() for c in cat_string.split(',')]

        self.df['category_list'] = self.df['categories'].apply(split_categories)

        # Get all unique categories
        all_categories = []
        for cat_list in self.df['category_list']:
            all_categories.extend(cat_list)

        unique_categories = list(set(all_categories))
        print(f"Total unique categories before normalization: {len(unique_categories)}")

        # Step 2: Normalize category names
        def normalize_category_name(cat):
            cat = cat.lower().strip()
            cat = re.sub(r'[^\w\s&]', '', cat)
            cat = re.sub(r'\s+', ' ', cat)
            return cat

        normalized_cats = {cat: normalize_category_name(cat) for cat in unique_categories}

        # Step 3: Create taxonomy and map categories
        category_taxonomy = self._create_category_taxonomy()

        def map_to_standard_categories(cat_list):
            standard_cats = set()

            for cat in cat_list:
                normalized = normalized_cats.get(cat, cat.lower())

                # Check against ALL taxonomy groups (not just first hit)
                matched = False
                for standard_cat, keywords in category_taxonomy.items():
                    for keyword in keywords:
                        if keyword in normalized:
                            standard_cats.add(standard_cat)
                            matched = True
                            break  # done with this taxonomy group, move to next

                # If no match in any group, keep original normalized category
                if not matched and normalized != 'uncategorized':
                    standard_cats.add(normalized)

            return list(standard_cats) if standard_cats else ['Uncategorized']

        self.df['categories_normalized'] = self.df['category_list'].apply(map_to_standard_categories)

        # Primary category: first taxonomy-matched category, or first overall
        taxonomy_groups = set(self._create_category_taxonomy().keys())

        def pick_primary(cat_list):
            # Prefer a recognized taxonomy group as primary
            for cat in cat_list:
                if cat in taxonomy_groups:
                    return cat
            return cat_list[0] if cat_list else 'Uncategorized'

        self.df['primary_category'] = self.df['categories_normalized'].apply(pick_primary)

        # Count results
        all_normalized = []
        for cat_list in self.df['categories_normalized']:
            all_normalized.extend(cat_list)

        unique_normalized = list(set(all_normalized))
        print(f"Unique categories after normalization: {len(unique_normalized)}")

        print("\nTop 20 normalized categories:")
        cat_counts = pd.Series(all_normalized).value_counts()
        print(cat_counts.head(20))

        self.cleaning_stats['categories_before'] = len(unique_categories)
        self.cleaning_stats['categories_after'] = len(unique_normalized)
        return self

    def _create_category_taxonomy(self):
        """
        Create standardized category taxonomy using grouping rules.
        Keywords use substring matching against normalized category names.
        """
        taxonomy = {
            'Food & Restaurants': [
                # General
                'restaurant', 'food', 'dining', 'eatery', 'cuisine', 'diner',
                'fast food', 'food court', 'food stand', 'food truck',
                # Cafe / Coffee / Drinks
                'cafe', 'coffee', 'tea', 'juice', 'smoothie', 'boba',
                'bar', 'pub', 'lounge', 'brewery', 'brewerie', 'beer',
                'wine', 'winer', 'cocktail', 'distillery', 'taproom',
                # Bakery / Sweets
                'bakery', 'bakerie', 'bakeri', 'donut', 'bagel', 'pastry',
                'dessert', 'ice cream', 'frozen yogurt', 'gelato', 'candy',
                'chocolate', 'cupcake', 'cookie',
                # Meal types
                'breakfast', 'brunch', 'buffet', 'caterer', 'catering',
                # Proteins / Specific foods
                'pizza', 'burger', 'sandwich', 'sub', 'hoagie', 'wrap',
                'steakhouse', 'steak', 'seafood', 'sushi', 'ramen', 'noodle',
                'chicken', 'wing', 'rib', 'bbq', 'barbeque', 'grill',
                'taco', 'burrito', 'cheesesteak', 'pretzel', 'hot dog',
                'soup', 'salad', 'deli', 'delis', 'meat shop',
                # Cuisines
                'chinese', 'italian', 'mexican', 'japanese', 'thai', 'indian',
                'korean', 'vietnamese', 'mediterranean', 'greek', 'turkish',
                'french', 'spanish', 'american', 'southern', 'cajun', 'creole',
                'ethiopian', 'latin', 'caribbean', 'cuban', 'peruvian',
                'middle eastern', 'lebanese', 'pakistani', 'halal',
                'asian fusion', 'pan asian', 'texmex', 'tex-mex',
                # Dietary
                'vegan', 'vegetarian', 'glutenfree', 'gluten-free', 'kosher',
                # Specialty
                'specialty food', 'ethnic food', 'health market',
                'cheese shop', 'fruit', 'veggie', 'organic',
            ],
            'Shopping & Retail': [
                'shopping', 'retail', 'store', 'boutique', 'mall', 'outlet',
                'clothing', 'fashion', 'apparel', 'wear', 'shoes', 'sneaker',
                'accessories', 'jewelry', 'jeweler', 'watch',
                'grocery', 'supermarket', 'convenience',
                'home & garden', 'home decor', 'furniture', 'flooring',
                'hardware', 'appliance', 'electronics', 'computer', 'phone',
                'mobile phone', 'shipping center', 'post',
                'flowers', 'florist', 'gift', 'cards',
                'books', 'magazine', 'mags', 'toy', 'hobby', 'craft',
                'sporting goods', 'outdoor', 'bicycle', 'bike',
                'music store', 'instrument', 'record',
                'antique', 'thrift', 'consignment', 'vintage',
                'optical', 'eyewear', 'glasses',
                'mattress', 'bedding', 'kitchen & bath', 'bath ',
                'building supplies', 'lumber', 'nurseries', 'garden center',
                'tobacco', 'vape', 'smoke shop',
                'interior design', 'shades', 'blinds', 'curtain',
                'bridal', 'wedding ', 'prom',
                'used ', 'second hand', 'pawn',
                'chocolatier', 'candy shop',
            ],
            'Health & Medical': [
                'health', 'medical', 'medicine', 'doctor', 'physician',
                'dentist', 'dental', 'orthodont', 'endodontist', 'periodontist',
                'oral surgeon', 'oral surgery',
                'hospital', 'clinic', 'urgent care', 'emergency',
                'pharmacy', 'drug store', 'prescription',
                'wellness', 'holistic', 'naturopath', 'acupuncture',
                'chiropractor', 'physical therapy', 'occupational therapy',
                'mental health', 'psychiatry', 'psychology', 'counseling',
                'optometrist', 'ophthalmolog', 'eye doctor',
                'dermatolog', 'skin care clinic',
                'pediatric', 'obgyn', 'gynecolog',
                'cardiolog', 'orthopedic', 'neurology',
                'lab', 'diagnostic', 'imaging', 'radiology',
                'hearing', 'audiology',
                'family practice', 'family medicine', 'primary care',
                'nutritionist', 'dietitian',
                'reflexology', 'weight loss', 'rehabilitation', 'rehab',
                'blood', 'dialysis', 'infusion',
            ],
            'Beauty & Spas': [
                'beauty', 'spa', 'salon', 'hair', 'nail', 'nails',
                'massage', 'skincare', 'skin care', 'facial',
                'cosmetic', 'makeup', 'makeup artist',
                'barber', 'barbershop', 'waxing', 'wax',
                'eyelash', 'lash', 'eyebrow', 'brow',
                'tanning', 'spray tan',
                'threading', 'microblading', 'tattoo', 'piercing',
                'blow dry', 'blowout',
            ],
            'Automotive': [
                'automotive', 'car ', 'auto ', 'auto repair', 'auto service',
                'mechanic', 'tire', 'wheel',
                'oil change', 'body shop', 'collision',
                'car wash', 'detailing', 'detail',
                'dealership', 'car dealer', 'used car',
                'towing', 'roadside',
                'parking', 'garage', 'gas station',
                'motorcycle', 'rv', 'truck rental',
                'transmission',
            ],
            'Home Services': [
                'home service', 'contractor', 'plumbing', 'plumber',
                'electrical', 'electrician', 'hvac', 'heating', 'cooling',
                'air conditioning', 'cleaning service', 'house cleaning',
                'home cleaning', 'office cleaning', 'janitorial',
                'landscaping', 'lawn', 'tree service', 'snow removal',
                'roofing', 'painting', 'painter', 'handyman',
                'renovation', 'remodel', 'construction', 'general contractor',
                'masonry', 'concrete', 'drywall', 'insulation',
                'pest control', 'exterminator',
                'moving', 'mover', 'storage', 'junk removal',
                'laundry', 'dry clean', 'alterations', 'sewing', 'tailor',
                'carpet', 'flooring installer', 'window',
                'locksmith', 'security',
                'apartment', 'property management',
                'water heater', 'drain', 'septic',
                'damage restoration', 'fire restoration', 'water damage',
            ],
            'Professional Services': [
                'professional service', 'lawyer', 'attorney', 'legal',
                'accountant', 'accounting', 'tax', 'cpa', 'bookkeeping',
                'financial', 'finance', 'investment', 'wealth management',
                'insurance', 'real estate', 'mortgage', 'title company',
                'consultant', 'consulting', 'staffing', 'recruiting',
                'marketing', 'advertising', 'printing', 'notary',
                'bank', 'credit union', 'atm', 'check cashing',
                'it service', 'tech support', 'software',
                'transportation', 'courier', 'freight', 'logistics',
                'telecom', 'telecommunication', 'internet provider',
                'funeral', 'cemetery', 'cremation',
                'photography', 'photographer', 'videography',
                'translation', 'interpreter',
                'street vendor', 'vendor',
                'event planning', 'wedding planning', 'party planning',
            ],
            'Entertainment & Arts': [
                'entertainment', 'arts', 'art ', 'theater', 'theatre',
                'cinema', 'movie', 'film',
                'museum', 'gallery', 'exhibit',
                'music', 'concert', 'live music', 'jazz', 'comedy',
                'nightlife', 'night club', 'lounge', 'club ',
                'casino', 'arcade', 'bowling', 'billiard', 'pool hall',
                'escape room', 'axe throwing', 'mini golf',
                'amusement', 'theme park', 'water park',
                'local flavor', 'landmark', 'attraction', 'tour',
                'festival', 'fair',
            ],
            'Active Life & Fitness': [
                'gym', 'fitness', 'yoga', 'pilates', 'barre', 'crossfit',
                'sports', 'recreation', 'recreational',
                'active life', 'active ',
                'park', 'trail', 'hiking', 'rock climbing', 'climbing',
                'golf', 'tennis', 'swimming', 'pool',
                'martial arts', 'karate', 'boxing', 'mma', 'jiu jitsu',
                'dance', 'dance studio',
                'cycling', 'spin', 'running',
                'trainer', 'personal trainer', 'coaching',
                'batting cage', 'sports complex',
                'skating', 'skate',
                'boot camp', 'bootcamp',
                'summer camp', 'camp ',
            ],
            'Hotels & Travel': [
                'hotel', 'motel', 'travel', 'lodging',
                'bed & breakfast', 'inn', 'hostel', 'resort',
                'vacation rental', 'airbnb',
                'airline', 'airport', 'car rental',
                'cruise', 'tour operator', 'travel agent',
            ],
            'Education': [
                'education', 'school', 'university', 'college',
                'tutoring', 'tutor', 'learning center',
                'training', 'classes', 'instruction',
                'daycare', 'day care', 'preschool', 'child care', 'childcare',
                'montessori', 'after school',
                'driving school', 'flight school',
                'art class', 'music lesson', 'language school',
            ],
            'Pets': [
                'pet', 'veterinarian', 'veterinary', 'vet ', 'animal',
                'dog', 'cat ', 'bird', 'fish ',
                'grooming', 'pet store', 'pet supply',
                'dog walker', 'dog training', 'pet sitting', 'kennel',
            ],
            'Religious Organizations': [
                'religious', 'religion', 'church', 'synagogue',
                'mosque', 'temple', 'cathedral', 'chapel',
                'spiritual', 'ministry', 'parish',
            ],
            'Local & Community Services': [
                'local service', 'community', 'nonprofit', 'charity',
                'government', 'public service', 'social service',
                'post office', 'library', 'community center',
            ],
        }

        return taxonomy

    def save_cleaned_data(self, output_file):
        """
        Save cleaned data to CSV

        Parameters:
        -----------
        output_file : str or Path
            Output file path
        """
        print("\n--- Saving Cleaned Data ---")

        # Select final columns (optimized - dropped redundant fields)
        # Dropped: 'categories' (redundant), 'valid_coordinates' (all True after cleaning)
        output_columns = [
            'business_id', 'name', 'address', 'city', 'state', 'postal_code',
            'latitude', 'longitude', 'stars', 'review_count', 'is_open',
            'primary_category', 'categories_normalized'
        ]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df[output_columns].to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Final dataset size: {len(self.df)} businesses")
        print(f"Final columns: {len(output_columns)}")

        # Save cleaning statistics
        stats_file = output_path.parent / 'yelp_business_cleaning_stats.txt'
        with open(stats_file, 'w') as f:
            f.write("YELP DATA CLEANING STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            for key, value in self.cleaning_stats.items():
                f.write(f"{key}: {value}\n")

        print(f"Cleaning statistics saved to: {stats_file}")
        return self

    def print_cleaning_summary(self):
        """Print summary of cleaning operations"""
        print("\n" + "=" * 80)
        print("CLEANING SUMMARY")
        print("=" * 80)

        for key, value in self.cleaning_stats.items():
            print(f"{key}: {value}")

        print("=" * 80)


def main():
    """Main execution function"""
    data_dir = Path(__file__).parent.parent.parent / "data"
    input_file = data_dir / "processed" / "yelp_philly_business_filtered.csv"
    output_file = data_dir / "processed" / "yelp_philly_business_clean.csv"

    cleaner = YelpDataCleaner(input_file)

    (cleaner
     .load_data()
     .standardize_column_names()
     .clean_location_fields()
     .handle_missing_values()
     .deduplicate_records()
     .normalize_categories()
     .save_cleaned_data(output_file)
     .print_cleaning_summary())

    print("\n✓ Yelp data cleaning complete!")
    print(f"✓ Final business dataset: {output_file}")


if __name__ == "__main__":
    main()
