class ActivityType:
    WALKING = 1
    DESCENDING_STAIRS = 2
    ASCENDING_STAIRS = 3
    DRIVING = 4
    CLAPPING = 77
    NON_STUDY_ACTIVITY = 99

    @staticmethod
    def get_activity_name(activity_id):
        activity_mapping = {
            ActivityType.WALKING: "Walking",
            ActivityType.DESCENDING_STAIRS: "Descending Stairs",
            ActivityType.ASCENDING_STAIRS: "Ascending Stairs",
            ActivityType.DRIVING: "Driving",
            ActivityType.CLAPPING: "Clapping",
            ActivityType.NON_STUDY_ACTIVITY: "Non-Study Activity"
        }
        return activity_mapping.get(activity_id, "Unknown")
