#!/usr/bin/env python3
"""
Debug the family survival rules to find all 18 passengers
"""

import pandas as pd

def load_data():
    try:
        train_df = pd.read_csv('titanic/train.csv')
        test_df = pd.read_csv('titanic/test.csv')
        return train_df, test_df
    except FileNotFoundError:
        return None, None

def debug_family_rules():
    train_df, test_df = load_data()
    if train_df is None:
        return
    
    # Expected passengers from the rules
    expected_masters = [956, 981, 1053, 1086, 1088, 1199, 1284, 1309]
    expected_females = [910, 925, 929, 1024, 1032, 1080, 1172, 1176, 1257, 1259]
    
    print("DEBUGGING FAMILY SURVIVAL RULES")
    print("=" * 50)
    
    # Check if all expected passengers exist in test set
    print("Expected Masters in test set:")
    for pid in expected_masters:
        if pid in test_df['PassengerId'].values:
            name = test_df[test_df['PassengerId'] == pid]['Name'].iloc[0]
            print(f"  ✓ {pid}: {name}")
        else:
            print(f"  ✗ {pid}: NOT FOUND")
    
    print("\nExpected Females in test set:")
    for pid in expected_females:
        if pid in test_df['PassengerId'].values:
            name = test_df[test_df['PassengerId'] == pid]['Name'].iloc[0]
            print(f"  ✓ {pid}: {name}")
        else:
            print(f"  ✗ {pid}: NOT FOUND")
    
    # Extract family info
    train_df['Surname'] = train_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    test_df['Surname'] = test_df['Name'].str.extract(r'([A-Za-z]+),', expand=False)
    test_df['Title'] = test_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    print(f"\n" + "=" * 50)
    print("ANALYZING FAMILY SURVIVAL PATTERNS")
    print("=" * 50)
    
    # Check each expected passenger's family
    all_expected = expected_masters + expected_females
    
    for pid in all_expected:
        if pid in test_df['PassengerId'].values:
            passenger = test_df[test_df['PassengerId'] == pid].iloc[0]
            surname = passenger['Surname']
            title = passenger['Title']
            
            print(f"\nPassenger {pid}: {passenger['Name']}")
            print(f"  Surname: {surname}, Title: {title}")
            
            # Find family in training data
            family_in_train = train_df[train_df['Surname'] == surname]
            if len(family_in_train) > 0:
                print(f"  Family in training: {len(family_in_train)} members")
                
                # Non-adult males (exclude Mr)
                non_adult_males = family_in_train[family_in_train['Title'] != 'Mr']
                if len(non_adult_males) > 0:
                    survival_rate = non_adult_males['Survived'].mean()
                    print(f"  Non-adult males: {len(non_adult_males)}, Survival rate: {survival_rate:.2f}")
                    
                    if survival_rate == 1.0:
                        print(f"  → Should predict LIVE (Rule 1)")
                    elif survival_rate == 0.0:
                        print(f"  → Should predict DIE (Rule 2)")
                    else:
                        print(f"  → Mixed survival, no rule applies")
                else:
                    print(f"  No non-adult males in family")
            else:
                print(f"  No family found in training data")

if __name__ == "__main__":
    debug_family_rules()